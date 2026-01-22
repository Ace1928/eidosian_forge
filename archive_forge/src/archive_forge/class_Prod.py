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
class Prod(Operation):

    def __init__(self, axis=None, keepdims=False, dtype=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.keepdims = keepdims
        self.dtype = dtype

    def call(self, x):
        return backend.numpy.prod(x, axis=self.axis, keepdims=self.keepdims, dtype=self.dtype)

    def compute_output_spec(self, x):
        if self.dtype is not None:
            dtype = self.dtype
        else:
            dtype = backend.result_type(x.dtype)
            if dtype == 'bool':
                dtype = 'int32'
            elif dtype in ('int8', 'int16'):
                dtype = 'int32'
            elif dtype in ('uint8', 'uint16'):
                dtype = 'uint32'
        if backend.backend() == 'torch' and dtype == 'uint32':
            dtype = 'int32'
        return KerasTensor(reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims), dtype=dtype)