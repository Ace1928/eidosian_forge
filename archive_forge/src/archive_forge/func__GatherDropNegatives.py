import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
def _GatherDropNegatives(params, ids, zero_clipped_indices=None, is_positive=None):
    """ Helper function for unsorted segment ops.

  Gathers params for
      positive segment ids and gathers 0 for inputs with negative segment id.
      Also returns the clipped indices and a boolean mask with the same shape
      as ids where a positive id is masked as true. With this, the latter two
      can be passed as arguments to this function to reuse them.
  """
    if zero_clipped_indices is None:
        zero_clipped_indices = math_ops.maximum(ids, array_ops.zeros_like(ids))
    gathered = array_ops.gather(params, zero_clipped_indices)
    if is_positive is None:
        is_positive = math_ops.greater_equal(ids, 0)
        is_positive_shape = array_ops.shape(is_positive)
        broadcastable_shape = array_ops.concat([is_positive_shape, array_ops.ones([array_ops.rank(gathered) - array_ops.rank(is_positive)], dtype=is_positive_shape.dtype)], axis=0)
        is_positive = array_ops.reshape(is_positive, broadcastable_shape)
        is_positive = is_positive & array_ops.ones_like(gathered, dtype=dtypes.bool)
    zero_slice = array_ops.zeros_like(gathered)
    return (array_ops.where_v2(is_positive, gathered, zero_slice), zero_clipped_indices, is_positive)