import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_string_ops import *
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _reduce_join_reduction_dims(x, axis):
    """Returns range(rank(x) - 1, 0, -1) if axis is None; or axis otherwise."""
    if axis is not None:
        return axis
    else:
        if x.get_shape().ndims is not None:
            return constant_op.constant(np.arange(x.get_shape().ndims - 1, -1, -1), dtype=dtypes.int32)
        return math_ops.range(array_ops.rank(x) - 1, -1, -1)