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
@tf_export('math.xdivy')
@dispatch.register_binary_elementwise_api
@dispatch.add_dispatch_support
def xdivy(x, y, name=None):
    """Computes `x / y`.

  Given `x` and `y`, computes `x / y`. This function safely returns
  zero when `x = 0`, no matter what the value of `y` is.

  Example:

  >>> tf.math.xdivy(1., 2.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
  >>> tf.math.xdivy(0., 1.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.math.xdivy(0., 0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=0.0>
  >>> tf.math.xdivy(1., 0.)
  <tf.Tensor: shape=(), dtype=float32, numpy=inf>

  Args:
    x: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    y: A `tf.Tensor` of type `half`, `float32`, `float64`, `complex64`,
      `complex128`
    name: A name for the operation (optional).

  Returns:
    `x / y`.
  """
    with ops.name_scope(name, 'xdivy', [x]):
        return gen_math_ops.xdivy(x, y)