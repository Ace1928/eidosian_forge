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
class Moveaxis(Operation):

    def __init__(self, source, destination):
        super().__init__()
        if isinstance(source, int):
            self.source = [source]
        else:
            self.source = source
        if isinstance(destination, int):
            self.destination = [destination]
        else:
            self.destination = destination
        if len(self.source) != len(self.destination):
            raise ValueError(f'`source` and `destination` arguments must have the same number of elements, but received `source={source}` and `destination={destination}`.')

    def call(self, x):
        return backend.numpy.moveaxis(x, self.source, self.destination)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        output_shape = [-1 for _ in range(len(x.shape))]
        for sc, dst in zip(self.source, self.destination):
            output_shape[dst] = x_shape[sc]
            x_shape[sc] = -1
        i, j = (0, 0)
        while i < len(output_shape):
            while i < len(output_shape) and output_shape[i] != -1:
                i += 1
            while j < len(output_shape) and x_shape[j] == -1:
                j += 1
            if i == len(output_shape):
                break
            output_shape[i] = x_shape[j]
            i += 1
            j += 1
        return KerasTensor(output_shape, dtype=x.dtype)