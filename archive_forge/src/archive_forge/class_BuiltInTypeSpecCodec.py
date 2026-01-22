import collections
import functools
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class BuiltInTypeSpecCodec:
    """Codec for built-in `TypeSpec` classes.

  Built-in TypeSpec's that do not require a custom codec implementation
  register themselves by instantiating this class and passing it to
  register_codec.

  Attributes:
    type_spec_class: The built-in TypeSpec class that the
      codec is instantiated for.
    type_spec_proto_enum: The proto enum value for the built-in TypeSpec class.
  """
    _BUILT_IN_TYPE_SPEC_PROTOS = []
    _BUILT_IN_TYPE_SPECS = []

    def __init__(self, type_spec_class, type_spec_proto_enum):
        if not issubclass(type_spec_class, internal.TypeSpec):
            raise ValueError(f"The type '{type_spec_class}' does not subclass tf.TypeSpec.")
        if type_spec_class in self._BUILT_IN_TYPE_SPECS:
            raise ValueError(f"The type '{type_spec_class}' already has an instantiated codec.")
        if type_spec_proto_enum in self._BUILT_IN_TYPE_SPEC_PROTOS:
            raise ValueError(f"The proto value '{type_spec_proto_enum}' is already registered.")
        if not isinstance(type_spec_proto_enum, int) or type_spec_proto_enum <= 0 or type_spec_proto_enum > 10:
            raise ValueError(f"The proto value '{type_spec_proto_enum}' is invalid.")
        self.type_spec_class = type_spec_class
        self.type_spec_proto_enum = type_spec_proto_enum
        self._BUILT_IN_TYPE_SPECS.append(type_spec_class)
        self._BUILT_IN_TYPE_SPEC_PROTOS.append(type_spec_proto_enum)

    def can_encode(self, pyobj):
        """Returns true if `pyobj` can be encoded as the built-in TypeSpec."""
        return isinstance(pyobj, self.type_spec_class)

    def do_encode(self, type_spec_value, encode_fn):
        """Returns an encoded proto for the given built-in TypeSpec."""
        type_state = type_spec_value._serialize()
        num_flat_components = len(nest.flatten(type_spec_value._component_specs, expand_composites=True))
        encoded_type_spec = struct_pb2.StructuredValue()
        encoded_type_spec.type_spec_value.CopyFrom(struct_pb2.TypeSpecProto(type_spec_class=self.type_spec_proto_enum, type_state=encode_fn(type_state), type_spec_class_name=self.type_spec_class.__name__, num_flat_components=num_flat_components))
        return encoded_type_spec

    def can_decode(self, value):
        """Returns true if `value` can be decoded into its built-in TypeSpec."""
        if value.HasField('type_spec_value'):
            type_spec_class_enum = value.type_spec_value.type_spec_class
            return type_spec_class_enum == self.type_spec_proto_enum
        return False

    def do_decode(self, value, decode_fn):
        """Returns the built in `TypeSpec` encoded by the proto `value`."""
        type_spec_proto = value.type_spec_value
        return self.type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))