import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class Cast(Operation):

    def __init__(self, dtype):
        super().__init__()
        self.dtype = backend.standardize_dtype(dtype)

    def call(self, x):
        return backend.core.cast(x, self.dtype)

    def compute_output_spec(self, x):
        return backend.KerasTensor(shape=x.shape, dtype=self.dtype)