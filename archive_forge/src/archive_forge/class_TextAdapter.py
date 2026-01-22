import numpy as np
import pandas as pd
import tensorflow as tf
from autokeras.engine import adapter as adapter_module
class TextAdapter(adapter_module.Adapter):

    def check(self, x):
        """Record any information needed by transform."""
        if not isinstance(x, (np.ndarray, tf.data.Dataset)):
            raise TypeError('Expect the data to TextInput to be numpy.ndarray or tf.data.Dataset, but got {type}.'.format(type=type(x)))