import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('unique_with_counts')
@dispatch.add_dispatch_support
def unique_with_counts(x, out_idx=dtypes.int32, name=None):
    """Finds unique elements in a 1-D tensor.

  See also `tf.unique`.

  This operation returns a tensor `y` containing all the unique elements
  of `x` sorted in the same order that they occur in `x`. This operation
  also returns a tensor `idx` the same size as `x` that contains the index
  of each value of `x` in the unique output `y`. Finally, it returns a
  third tensor `count` that contains the count of each element of `y`
  in `x`. In other words:

    y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]

  Example usage:

  >>> x = tf.constant([1, 1, 2, 4, 4, 4, 7, 8, 8])
  >>> y, idx, count = unique_with_counts(x)
  >>> y
  <tf.Tensor: id=8, shape=(5,), dtype=int32,
  numpy=array([1, 2, 4, 7, 8], dtype=int32)>
  >>> idx
  <tf.Tensor: id=9, shape=(9,), dtype=int32,
  numpy=array([0, 0, 1, 2, 2, 2, 3, 4, 4], dtype=int32)>
  >>> count
  <tf.Tensor: id=10, shape=(5,), dtype=int32,
  numpy=array([2, 1, 3, 1, 2], dtype=int32)>

  Args:
    x: A Tensor. 1-D.
    out_idx: An optional tf.DType from: tf.int32, tf.int64. Defaults to
      tf.int32.
    name: A name for the operation (optional).

  Returns:
    A tuple of Tensor objects (y, idx, count).
      y: A Tensor. Has the same type as x.
      idx: A Tensor of type out_idx.
      count: A Tensor of type out_idx.

  """
    return gen_array_ops.unique_with_counts(x, out_idx, name)