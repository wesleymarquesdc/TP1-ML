#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score


# DADOS - OBTENÇÃO E TRATAMENTO

# In[37]:


dataset = np.loadtxt('data_tp1', delimiter=',')
X = dataset[:, 1:]
y = dataset[:, 0]


# In[38]:


plt.hist(y, alpha=0.5)
plt.show()


# In[39]:


# criação do test set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# criação do validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)


# In[40]:


plt.hist(y_train, alpha=0.5, label='Train Set')
plt.hist(y_test, alpha=0.7, label='Test Set')
plt.hist(y_valid, alpha=0.6, label='Validation Set')
plt.legend()
plt.show()


# In[41]:


some_digit = X_train[2]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap='binary')
plt.show()


# In[42]:


y_train[2]


# In[43]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)


# MODELO - CRIAÇÃO E TREINAMENTO

# In[44]:


def create_model(n_neurons, learning_rate):
    # definição
    model = keras.models.Sequential([
        keras.layers.Input(shape=(784,)), # input layer
        keras.layers.Dense(n_neurons, activation='sigmoid'), # hidden layer
        keras.layers.Dense(10, activation='softmax') # output layer
    ])
    # compilação
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model


# In[45]:


def train_model(X_train, X_valid, y_train, y_valid, model, batch_size):
    history = model.fit(X_train, y_train, 
                        epochs=200, 
                        batch_size=batch_size, 
                        validation_data=(X_valid, y_valid))
    return history


# In[46]:


def save_models(param_distribs, X_train, X_valid, y_train, y_valid):
    for n_neurons in param_distribs['n_neurons']:
        for learning_rate in param_distribs['learning_rate']:
            for batch_size in param_distribs['batch_size']:
                
                # cria e treina cada modelo
                model = create_model(n_neurons, learning_rate)
                history = train_model(X_train, X_valid, y_train, y_valid, model, batch_size)
                
                # salva o modelo e seu histórico
                folder = f'saved_models_2/model_{n_neurons}_{learning_rate}_{batch_size}'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                model_filename = f'{folder}/model.joblib'
                history_filename = f'{folder}/history.joblib'
                joblib.dump(model, model_filename)
                joblib.dump(history.history, history_filename)


# In[47]:


def load_models(param_distribs):
    models = []
    for n_neurons in param_distribs['n_neurons']:
        for learning_rate in param_distribs['learning_rate']:
            for batch_size in param_distribs['batch_size']:
                
                folder = f'saved_models_2/model_{n_neurons}_{learning_rate}_{batch_size}'
                # carrega o modelo e seu histórico
                model_filename = f'{folder}/model.joblib'
                history_filename = f'{folder}/history.joblib'
                model = joblib.load(model_filename)
                history = joblib.load(history_filename)
                models.append({'model': model, 'history': history, 'n_neurons': n_neurons, 'learning_rate': learning_rate, 'batch_size': batch_size})
    
    return models


# In[48]:


def plot(n_neurons, models, param_distribs):
    fig, axs = plt.subplots(len(param_distribs['batch_size']), len(param_distribs['learning_rate']), figsize=(20, 15))
    fig.suptitle(f'Number of Neurons: {n_neurons}', fontsize=20, fontweight='bold')
    
    for i, batch_size in enumerate(param_distribs['batch_size']):
        for j, learning_rate in enumerate(param_distribs['learning_rate']):
            for model in models:
                if(model['n_neurons']==n_neurons and model['batch_size']==batch_size and model['learning_rate']==learning_rate):
                    # plota o erro empírico
                    axs[i, j].plot(model['history']['loss'], label=f'loss (lr={learning_rate})')
                    axs[i, j].plot(model['history']['val_loss'], label=f'val_loss (lr={learning_rate})')
                    
                    if batch_size == param_distribs['batch_size'][0]:
                        axs[i, 1].set_title(f'Gradient Descent', fontsize=16, fontweight='bold')
                    if batch_size == param_distribs['batch_size'][1]:
                        axs[i, 1].set_title(f'Mini-Batch: 50', fontsize=16, fontweight='bold')
                    if batch_size == param_distribs['batch_size'][2]:
                        axs[i, 1].set_title(f'Mini-Batch: 10', fontsize=16, fontweight='bold')
                    if batch_size == param_distribs['batch_size'][3]:
                        axs[i, 1].set_title(f'Stochastic Gradient Descent', fontsize=16, fontweight='bold')
                    axs[i, j].set_ylabel('Empirical Error')
                    axs[i, j].set_xlabel('Epochs')
                    axs[i, j].legend(loc='upper right')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


# In[49]:


param_distribs = {
    'n_neurons': [25, 50, 100],
    'learning_rate': [0.5, 1.0, 10.0],
    'batch_size': [len(X_train), 50, 10, 1]
}


# In[50]:


save_models(param_distribs, X_train, X_valid, y_train, y_valid)


