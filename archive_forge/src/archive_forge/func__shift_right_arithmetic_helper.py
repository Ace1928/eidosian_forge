from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils
def _shift_right_arithmetic_helper(x, y, name=None):
    """Performs an integer right arithmetic shift irrespective of input type."""
    assert y.dtype == x.dtype
    dtype = x.dtype
    unsigned = dtype in _UNSIGNED_TO_SIGNED_TABLE
    if unsigned:
        signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[dtype]
        x = math_ops.cast(x, signed_dtype)
        y = math_ops.cast(y, signed_dtype)
    output = bitwise_ops.right_shift(x, y, name=name)
    if unsigned:
        output = math_ops.cast(output, dtype)
    return output