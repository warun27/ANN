# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 03:48:37 2020

@author: shara
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import calssification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


forest = pd.read_csv("F:\\Warun\\DS Assignments\\DS Assignments\\Neural_Network\\forestfires.csv")
forest.describe()
forest.info()
plt.style.use('seaborn')
forest1.hist(bins= 30, figsize=(20,15))
forest1.plot(kind='density', subplots=True, layout=(8,4), sharex=False, sharey=False)
forest1.plot(kind='box', subplots=True, layout=(6,5), sharex=False, sharey=False)

scatter_matrix(forest1)


forest1 = forest.drop(columns=['month', 'day'])
forest1.loc[forest1.size_category=="small", "size_category"] = 0
forest1.loc[forest1.size_category=="large", "size_category"] = 1
corr_forest = forest1.corr()
print(forest1["size_category"].dtypes)

x = forest1.drop(columns = ["size_category"])
y = forest1["size_category"]
forest1.dtypes
forest1["size_category"].value_counts()

# pip install imbalanced-learn
# import imblearn
# from imblearn.over_sampling import SMOTE
# smote = SMOTE()

pip install keras
pip install tensorflow
from keras.utils import to_categorical
y = y.astype("category")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Embedding
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(units = 28, kernel_initializer = "uniform", activation = "relu", input_dim = 28))
model.add(Dense(units = 100, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(units = 100, kernel_initializer = "uniform", activation = "relu"))
model.add(Dense(units = 2, kernel_initializer = "uniform", activation = "softmax"))
model.summary()
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
model.fit(np.array(x_train_scaled), np.array(y_train),validation_data = (x_test_scaled, y_test), batch_size = 10, epochs = 500)
y_test_pred = model.predict_classes(x_test_scaled)
y_train_pred = model.predict_classes(x_train_scaled)
print(confusion_matrix(y_test,y_test_pred))
np.mean(y_test==y_test_pred)
np.mean(y_train==y_train_pred)

