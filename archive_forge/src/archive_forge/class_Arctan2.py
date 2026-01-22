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
class Arctan2(Operation):

    def call(self, x1, x2):
        return backend.numpy.arctan2(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = getattr(x1, 'shape', [])
        x2_shape = getattr(x2, 'shape', [])
        outputs_shape = broadcast_shapes(x1_shape, x2_shape)
        x1_dtype = backend.standardize_dtype(getattr(x1, 'dtype', backend.floatx()))
        x2_dtype = backend.standardize_dtype(getattr(x2, 'dtype', backend.floatx()))
        dtype = dtypes.result_type(x1_dtype, x2_dtype, float)
        return KerasTensor(outputs_shape, dtype=dtype)