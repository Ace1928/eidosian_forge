import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import autokeras as ak
from benchmark.experiments import experiment
class Wine(StructuredDataClassifierExperiment):

    def __init__(self):
        super().__init__(name='Wine')

    @staticmethod
    def load_data():
        DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        dataset = tf.keras.utils.get_file('wine.csv', DATASET_URL)
        data = pd.read_csv(dataset, header=None).sample(frac=1, random_state=5)
        split_length = int(data.shape[0] * 0.8)
        return ((data.iloc[:split_length, 1:], data.iloc[:split_length, 0]), (data.iloc[split_length:, 1:], data.iloc[split_length:, 0]))