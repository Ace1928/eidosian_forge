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
class Repeat(Operation):

    def __init__(self, repeats, axis=None):
        super().__init__()
        self.axis = axis
        self.repeats = repeats

    def call(self, x):
        return backend.numpy.repeat(x, self.repeats, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        if self.axis is None:
            if None in x_shape:
                return KerasTensor([None], dtype=x.dtype)
            x_flatten_size = int(np.prod(x_shape))
            if isinstance(self.repeats, int):
                output_shape = [x_flatten_size * self.repeats]
            else:
                output_shape = [int(np.sum(self.repeats))]
            return KerasTensor(output_shape, dtype=x.dtype)
        size_on_ax = x_shape[self.axis]
        output_shape = x_shape
        if isinstance(self.repeats, int):
            if size_on_ax is None:
                output_shape[self.axis] = None
            else:
                output_shape[self.axis] = size_on_ax * self.repeats
        else:
            output_shape[self.axis] = int(np.sum(self.repeats))
        return KerasTensor(output_shape, dtype=x.dtype)