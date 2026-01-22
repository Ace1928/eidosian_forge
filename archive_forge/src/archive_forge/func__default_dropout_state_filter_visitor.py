import hashlib
import numbers
import sys
import types as python_types
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
def _default_dropout_state_filter_visitor(substate):
    from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import LSTMStateTuple
    if isinstance(substate, LSTMStateTuple):
        return LSTMStateTuple(c=False, h=True)
    elif isinstance(substate, tensor_array_ops.TensorArray):
        return False
    return True