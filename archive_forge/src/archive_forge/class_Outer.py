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
class Outer(Operation):

    def call(self, x1, x2):
        return backend.numpy.outer(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, 'shape', [1])
        x2_shape = getattr(x2, 'shape', [1])
        if None in x1_shape:
            x1_flatten_shape = None
        else:
            x1_flatten_shape = int(np.prod(x1_shape))
        if None in x2_shape:
            x2_flatten_shape = None
        else:
            x2_flatten_shape = int(np.prod(x2_shape))
        output_shape = [x1_flatten_shape, x2_flatten_shape]
        output_dtype = backend.result_type(getattr(x1, 'dtype', type(x1)), getattr(x2, 'dtype', type(x2)))
        return KerasTensor(output_shape, dtype=output_dtype)