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
class Cumsum(Operation):

    def __init__(self, axis=None, dtype=None):
        super().__init__()
        self.axis = axis
        self.dtype = dtype

    def call(self, x):
        return backend.numpy.cumsum(x, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.axis is None:
            if None in x.shape:
                output_shape = (None,)
            else:
                output_shape = (int(np.prod(x.shape)),)
        else:
            output_shape = x.shape
        output_dtype = backend.standardize_dtype(self.dtype or x.dtype)
        if output_dtype == 'bool':
            output_dtype = 'int32'
        return KerasTensor(output_shape, output_dtype)