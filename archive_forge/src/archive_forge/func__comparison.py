import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
def _comparison(tf_fun, x1, x2, cast_bool_to_int=False):
    """Helper function for comparision."""
    dtype = np_utils.result_type(x1, x2)
    x1 = np_array_ops.array(x1, dtype=dtype)
    x2 = np_array_ops.array(x2, dtype=dtype)
    if cast_bool_to_int and x1.dtype == dtypes.bool:
        x1 = math_ops.cast(x1, dtypes.int32)
        x2 = math_ops.cast(x2, dtypes.int32)
    return tf_fun(x1, x2)