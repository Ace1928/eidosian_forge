from enum import EnumMeta
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf.internal.enum_type_wrapper import EnumTypeWrapper
from proto.primitives import ProtoType
class MapField(Field):
    """A representation of a map field in protocol buffers."""

    def __init__(self, key_type, value_type, *, number: int, message=None, enum=None):
        super().__init__(value_type, number=number, message=message, enum=enum)
        self.map_key_type = key_type