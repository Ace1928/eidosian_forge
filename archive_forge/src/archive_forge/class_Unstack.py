import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class Unstack(Operation):

    def __init__(self, num=None, axis=0):
        super().__init__()
        self.num = num
        self.axis = axis

    def call(self, x):
        return backend.core.unstack(x, self.num, self.axis)

    def compute_output_spec(self, x):
        axis = self.axis
        if axis < 0:
            axis = len(x.shape) + axis
        output_shapes = x.shape[:axis] + x.shape[axis + 1:]
        num = self.num
        if num is None:
            num = x.shape[axis]
        if num is None:
            raise ValueError(f'Cannot infer argument `num` from shape {x.shape}. Either provide a tensor with a concrete shape in the `axis` dimension or explicitly pass the `num` argument.')
        output = [KerasTensor(shape=output_shapes, dtype=x.dtype) for _ in range(num)]
        return output