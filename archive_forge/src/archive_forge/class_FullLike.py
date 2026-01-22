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
class FullLike(Operation):

    def call(self, x, fill_value, dtype=None):
        return backend.numpy.full_like(x, fill_value, dtype=dtype)

    def compute_output_spec(self, x, fill_value, dtype=None):
        dtype = dtype or x.dtype
        return KerasTensor(x.shape, dtype=dtype)