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
class Vdot(Operation):

    def call(self, x1, x2):
        return backend.numpy.vdot(x1, x2)

    def compute_output_spec(self, x1, x2):
        dtype = dtypes.result_type(getattr(x1, 'dtype', type(x1)), getattr(x2, 'dtype', type(x2)))
        return KerasTensor([], dtype=dtype)