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
class Squeeze(Operation):

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x):
        return backend.numpy.squeeze(x, axis=self.axis)

    def compute_output_spec(self, x):
        input_shape = list(x.shape)
        sparse = getattr(x, 'sparse', False)
        if self.axis is None:
            output_shape = list(filter(1 .__ne__, input_shape))
            return KerasTensor(output_shape, dtype=x.dtype, sparse=sparse)
        else:
            if input_shape[self.axis] != 1:
                raise ValueError(f'Cannot squeeze axis {self.axis}, because the dimension is not 1.')
            del input_shape[self.axis]
            return KerasTensor(input_shape, dtype=x.dtype, sparse=sparse)