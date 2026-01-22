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
class ExpandDims(Operation):

    def __init__(self, axis):
        super().__init__()
        if isinstance(axis, list):
            raise ValueError(f'The `axis` argument to `expand_dims` should be an integer, but received a list: {axis}.')
        self.axis = axis

    def call(self, x):
        return backend.numpy.expand_dims(x, self.axis)

    def compute_output_spec(self, x):
        output_shape = operation_utils.compute_expand_dims_output_shape(x.shape, self.axis)
        sparse = getattr(x, 'sparse', False)
        return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)