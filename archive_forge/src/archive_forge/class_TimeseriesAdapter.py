import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module
class TimeseriesAdapter(adapter_module.Adapter):

    def __init__(self, lookback=None, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (pd.DataFrame, np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data in TimeseriesInput to be numpy.ndarray or tf.data.Dataset or pd.DataFrame, but got {type}.'.format(type=type(x)))

    def convert_to_dataset(self, dataset, batch_size):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.values
        return super().convert_to_dataset(dataset, batch_size)