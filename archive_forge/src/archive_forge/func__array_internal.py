import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _array_internal(val, dtype=None, copy=True, ndmin=0):
    """Main implementation of np.array()."""
    result_t = val
    if not isinstance(result_t, tensor_lib.Tensor):
        dtype = np_utils.result_type_unary(result_t, dtype)
        result_t = np_arrays.convert_to_tensor(result_t, dtype_hint=dtype)
        result_t = math_ops.cast(result_t, dtype=dtype)
    elif dtype:
        result_t = math_ops.cast(result_t, dtype)
    if copy:
        result_t = array_ops.identity(result_t)
    max_ndmin = 32
    if ndmin > max_ndmin:
        raise ValueError(f'ndmin bigger than allowable number of dimensions: {max_ndmin}.')
    if ndmin == 0:
        return result_t
    ndims = array_ops.rank(result_t)

    def true_fn():
        old_shape = array_ops.shape(result_t)
        new_shape = array_ops.concat([array_ops.ones(ndmin - ndims, dtypes.int32), old_shape], axis=0)
        return array_ops.reshape(result_t, new_shape)
    result_t = np_utils.cond(np_utils.greater(ndmin, ndims), true_fn, lambda: result_t)
    return result_t