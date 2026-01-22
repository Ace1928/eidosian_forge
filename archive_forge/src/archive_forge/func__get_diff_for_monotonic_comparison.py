import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _get_diff_for_monotonic_comparison(x):
    """Gets the difference x[1:] - x[:-1]."""
    x = array_ops.reshape(x, [-1])
    if not is_numeric_tensor(x):
        raise TypeError('Expected x to be numeric, instead found: %s' % x)
    is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
    short_result = lambda: ops.convert_to_tensor([], dtype=x.dtype)
    s_len = array_ops.shape(x) - 1
    diff = lambda: array_ops.strided_slice(x, [1], [1] + s_len) - array_ops.strided_slice(x, [0], s_len)
    return cond.cond(is_shorter_than_two, short_result, diff)