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
@tf_export.tf_export('experimental.numpy.true_divide', v1=[])
@np_utils.np_doc('true_divide')
def true_divide(x1, x2):

    def _avoid_float64(x1, x2):
        if x1.dtype == x2.dtype and x1.dtype in (dtypes.int32, dtypes.int64):
            x1 = math_ops.cast(x1, dtype=dtypes.float32)
            x2 = math_ops.cast(x2, dtype=dtypes.float32)
        return (x1, x2)

    def f(x1, x2):
        if x1.dtype == dtypes.bool:
            assert x2.dtype == dtypes.bool
            float_ = np_utils.result_type(float)
            x1 = math_ops.cast(x1, float_)
            x2 = math_ops.cast(x2, float_)
        if not np_dtypes.is_allow_float64():
            x1, x2 = _avoid_float64(x1, x2)
        return math_ops.truediv(x1, x2)
    return _bin_op(f, x1, x2)