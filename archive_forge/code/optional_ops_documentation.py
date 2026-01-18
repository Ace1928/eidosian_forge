import abc
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Returns a `tf.experimental.Optional` that wraps the given value.

    >>> optional = tf.experimental.Optional.from_value(42)
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)

    Args:
      value: A value to wrap. The value must be convertible to `tf.Tensor` or
        `tf.CompositeTensor`.

    Returns:
      A `tf.experimental.Optional` that wraps `value`.
    