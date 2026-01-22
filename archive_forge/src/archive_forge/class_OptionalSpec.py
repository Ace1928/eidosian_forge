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
@tf_export('OptionalSpec', v1=['OptionalSpec', 'data.experimental.OptionalStructure'])
class OptionalSpec(type_spec.TypeSpec):
    """Type specification for `tf.experimental.Optional`.

  For instance, `tf.OptionalSpec` can be used to define a tf.function that takes
  `tf.experimental.Optional` as an input argument:

  >>> @tf.function(input_signature=[tf.OptionalSpec(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))])
  ... def maybe_square(optional):
  ...   if optional.has_value():
  ...     x = optional.get_value()
  ...     return x * x
  ...   return -1
  >>> optional = tf.experimental.Optional.from_value(5)
  >>> print(maybe_square(optional))
  tf.Tensor(25, shape=(), dtype=int32)

  Attributes:
    element_spec: A (nested) structure of `TypeSpec` objects that represents the
      type specification of the optional element.
  """
    __slots__ = ['_element_spec']

    def __init__(self, element_spec):
        super().__init__()
        self._element_spec = element_spec

    @property
    def value_type(self):
        return _OptionalImpl

    def _serialize(self):
        return (self._element_spec,)

    @property
    def _component_specs(self):
        return [tensor_spec.TensorSpec((), dtypes.variant)]

    def _to_components(self, value):
        return [value._variant_tensor]

    def _from_components(self, flat_value):
        return _OptionalImpl(flat_value[0], self._element_spec)

    @staticmethod
    def from_value(value):
        return OptionalSpec(value.element_spec)

    def _to_legacy_output_types(self):
        return self

    def _to_legacy_output_shapes(self):
        return self

    def _to_legacy_output_classes(self):
        return self