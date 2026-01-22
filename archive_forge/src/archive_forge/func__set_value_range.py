import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _set_value_range(self, value_range):
    if not isinstance(value_range, (tuple, list)):
        raise ValueError(self.value_range_VALIDATION_ERROR + f'Received: value_range={value_range}')
    if len(value_range) != 2:
        raise ValueError(self.value_range_VALIDATION_ERROR + f'Received: value_range={value_range}')
    self.value_range = sorted(value_range)