import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module
class HeadAdapter(adapter_module.Adapter):

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def check(self, dataset):
        supported_types = (tf.data.Dataset, np.ndarray, pd.DataFrame, pd.Series)
        if not isinstance(dataset, supported_types):
            raise TypeError('Expect the target data of {name} to be tf.data.Dataset, np.ndarray, pd.DataFrame or pd.Series, but got {type}.'.format(name=self.name, type=type(dataset)))

    def convert_to_dataset(self, dataset, batch_size):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        if isinstance(dataset, pd.Series):
            dataset = dataset.values
        return super().convert_to_dataset(dataset, batch_size)