import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.random.seed_generator import SeedGenerator
def _set_factor(self, factor):
    if isinstance(factor, (tuple, list)):
        if len(factor) != 2:
            raise ValueError(self._FACTOR_VALIDATION_ERROR + f'Received: factor={factor}')
        self._check_factor_range(factor[0])
        self._check_factor_range(factor[1])
        self._factor = sorted(factor)
    elif isinstance(factor, (int, float)):
        self._check_factor_range(factor)
        factor = abs(factor)
        self._factor = [-factor, factor]
    else:
        raise ValueError(self._FACTOR_VALIDATION_ERROR + f'Received: factor={factor}')