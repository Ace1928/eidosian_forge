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
@tf_export('math.argmin', 'argmin', v1=[])
@dispatch.add_dispatch_support
def argmin_v2(input, axis=None, output_type=dtypes.int64, name=None):
    """Returns the index with the smallest value across axes of a tensor.

  Returns the smallest index in case of ties.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,
      `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`,
      `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to
      `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.

  Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmin(input = a)
  c = tf.keras.backend.eval(b)
  # c = 0
  # here a[0] = 1 which is the smallest element of a across axis 0
  ```
  """
    if axis is None:
        axis = 0
    return gen_math_ops.arg_min(input, axis, name=name, output_type=output_type)