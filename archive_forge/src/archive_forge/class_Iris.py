import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import autokeras as ak
from benchmark.experiments import experiment
class Iris(StructuredDataClassifierExperiment):

    def __init__(self):
        super().__init__(name='Iris')

    @staticmethod
    def load_data():
        TRAIN_DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
        x_train = tf.keras.utils.get_file('iris_train.csv', TRAIN_DATA_URL)
        TEST_DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
        x_test = tf.keras.utils.get_file('iris_test.csv', TEST_DATA_URL)
        return ((x_train, 'virginica'), (x_test, 'virginica'))