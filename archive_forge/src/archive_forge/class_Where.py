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
class Where(Operation):

    def call(self, condition, x1=None, x2=None):
        return backend.numpy.where(condition, x1, x2)

    def compute_output_spec(self, condition, x1, x2):
        condition_shape = getattr(condition, 'shape', [])
        x1_shape = getattr(x1, 'shape', [])
        x2_shape = getattr(x2, 'shape', [])
        output_shape = broadcast_shapes(condition_shape, x1_shape)
        output_shape = broadcast_shapes(output_shape, x2_shape)
        output_dtype = dtypes.result_type(getattr(x1, 'dtype', type(x1) if x1 is not None else 'int'), getattr(x2, 'dtype', type(x2) if x2 is not None else 'int'))
        return KerasTensor(output_shape, dtype=output_dtype)