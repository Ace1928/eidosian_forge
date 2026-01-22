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
class Take(Operation):

    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def call(self, x, indices):
        return backend.numpy.take(x, indices, axis=self.axis)

    def compute_output_spec(self, x, indices):
        x_shape = list(x.shape)
        if isinstance(indices, KerasTensor):
            indices_shape = list(indices.shape)
        else:
            indices_shape = list(getattr(np.array(indices), 'shape', []))
        if self.axis is None:
            return KerasTensor(indices_shape, dtype=x.dtype)
        axis = len(x_shape) + self.axis if self.axis < 0 else self.axis
        output_shape = x_shape[:axis] + indices_shape + x_shape[axis + 1:]
        return KerasTensor(output_shape, dtype=x.dtype)