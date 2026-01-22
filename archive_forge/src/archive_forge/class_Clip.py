import builtins
import re
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import dtypes
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape
class Clip(Operation):

    def __init__(self, x_min, x_max):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

    def call(self, x):
        return backend.numpy.clip(x, self.x_min, self.x_max)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(x.dtype)
        if dtype == 'bool':
            dtype = 'int32'
        return KerasTensor(x.shape, dtype=dtype)