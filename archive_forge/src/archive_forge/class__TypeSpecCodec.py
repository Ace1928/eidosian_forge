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
class _TypeSpecCodec:
    """Codec for `tf.TypeSpec`."""

    def can_encode(self, pyobj):
        """Returns true if `pyobj` can be encoded as a TypeSpec."""
        if isinstance(pyobj, internal.TypeSpec):
            try:
                type_spec_registry.get_name(type(pyobj))
                return True
            except ValueError:
                return False
        return False

    def do_encode(self, type_spec_value, encode_fn):
        """Returns an encoded proto for the given `tf.TypeSpec`."""
        type_spec_class_name = type_spec_registry.get_name(type(type_spec_value))
        type_spec_class = struct_pb2.TypeSpecProto.REGISTERED_TYPE_SPEC
        warnings.warn('Encoding a StructuredValue with type %s; loading this StructuredValue will require that this type be imported and registered.' % type_spec_class_name)
        type_state = type_spec_value._serialize()
        num_flat_components = len(nest.flatten(type_spec_value._component_specs, expand_composites=True))
        encoded_type_spec = struct_pb2.StructuredValue()
        encoded_type_spec.type_spec_value.CopyFrom(struct_pb2.TypeSpecProto(type_spec_class=type_spec_class, type_state=encode_fn(type_state), type_spec_class_name=type_spec_class_name, num_flat_components=num_flat_components))
        return encoded_type_spec

    def can_decode(self, value):
        """Returns true if `value` can be decoded into a `tf.TypeSpec`."""
        return value.HasField('type_spec_value')

    def do_decode(self, value, decode_fn):
        """Returns the `tf.TypeSpec` encoded by the proto `value`."""
        type_spec_proto = value.type_spec_value
        type_spec_class_enum = type_spec_proto.type_spec_class
        class_name = type_spec_proto.type_spec_class_name
        if type_spec_class_enum == struct_pb2.TypeSpecProto.REGISTERED_TYPE_SPEC:
            try:
                type_spec_class = type_spec_registry.lookup(class_name)
            except ValueError as e:
                raise ValueError(f"The type '{class_name}' has not been registered.  It must be registered before you load this object (typically by importing its module).") from e
        else:
            raise ValueError(f"The type '{class_name}' is not supported by this version of TensorFlow. (The object you are loading must have been created with a newer version of TensorFlow.)")
        return type_spec_class._deserialize(decode_fn(type_spec_proto.type_state))