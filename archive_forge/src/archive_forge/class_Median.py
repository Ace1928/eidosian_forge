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
class Median(Operation):

    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return backend.numpy.median(x, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        output_shape = reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims)
        if backend.standardize_dtype(x.dtype) == 'int64':
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(x.dtype, float)
        return KerasTensor(output_shape, dtype=dtype)