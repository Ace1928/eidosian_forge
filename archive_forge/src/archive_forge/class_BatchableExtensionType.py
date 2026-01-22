import abc
import typing
import warnings
import typing_extensions
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type_field
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.BatchableExtensionType')
class BatchableExtensionType(ExtensionType):
    """An ExtensionType that can be batched and unbatched.

  `BatchableExtensionType`s can be used with APIs that require batching or
  unbatching, including `Keras`, `tf.data.Dataset`, and `tf.map_fn`.  E.g.:

  >>> class Vehicle(tf.experimental.BatchableExtensionType):
  ...   top_speed: tf.Tensor
  ...   mpg: tf.Tensor
  >>> batch = Vehicle([120, 150, 80], [30, 40, 12])
  >>> tf.map_fn(lambda vehicle: vehicle.top_speed * vehicle.mpg, batch,
  ...           fn_output_signature=tf.int32).numpy()
  array([3600, 6000,  960], dtype=int32)

  An `ExtensionTypeBatchEncoder` is used by these APIs to encode `ExtensionType`
  values. The default encoder assumes that values can be stacked, unstacked, or
  concatenated by simply stacking, unstacking, or concatenating every nested
  `Tensor`, `ExtensionType`, `CompositeTensor`, or `TensorShape` field.
  Extension types where this is not the case will need to override
  `__batch_encoder__` with a custom `ExtensionTypeBatchEncoder`.  See
  `tf.experimental.ExtensionTypeBatchEncoder` for more details.
  """
    _tf_extension_type_do_not_transform_this_class = True