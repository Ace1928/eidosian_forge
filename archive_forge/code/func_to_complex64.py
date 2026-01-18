import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['to_complex64'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(date=None, instructions='Use `tf.cast` instead.')
def to_complex64(x, name='ToComplex64'):
    """Casts a tensor to type `complex64`.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
    type `complex64`.

  Raises:
    TypeError: If `x` cannot be cast to the `complex64`.

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but has an exact replacement
  `tf.cast(..., tf.complex64)`. There are no further issues with eager execution
  or tf.function.

  Before:

  >>> tf.compat.v1.to_complex64(tf.constant(1. + 2.j, dtype=tf.complex128))
  <tf.Tensor: shape=(), dtype=complex64, numpy=(1+2j)>

  After:

  >>> tf.cast(tf.constant(1. + 2.j, dtype=tf.complex128), tf.complex64)
  <tf.Tensor: shape=(), dtype=complex64, numpy=(1+2j)>

  @end_compatibility

  """
    return cast(x, dtypes.complex64, name=name)