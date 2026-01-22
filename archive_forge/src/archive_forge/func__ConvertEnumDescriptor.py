import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _ConvertEnumDescriptor(self, enum_proto, package=None, file_desc=None, containing_type=None, scope=None, top_level=False):
    """Make a protobuf EnumDescriptor given an EnumDescriptorProto protobuf.

    Args:
      enum_proto: The descriptor_pb2.EnumDescriptorProto protobuf message.
      package: Optional package name for the new message EnumDescriptor.
      file_desc: The file containing the enum descriptor.
      containing_type: The type containing this enum.
      scope: Scope containing available types.
      top_level: If True, the enum is a top level symbol. If False, the enum
          is defined inside a message.

    Returns:
      The added descriptor
    """
    if package:
        enum_name = '.'.join((package, enum_proto.name))
    else:
        enum_name = enum_proto.name
    if file_desc is None:
        file_name = None
    else:
        file_name = file_desc.name
    values = [self._MakeEnumValueDescriptor(value, index) for index, value in enumerate(enum_proto.value)]
    desc = descriptor.EnumDescriptor(name=enum_proto.name, full_name=enum_name, filename=file_name, file=file_desc, values=values, containing_type=containing_type, options=_OptionsOrNone(enum_proto), create_key=descriptor._internal_create_key)
    scope['.%s' % enum_name] = desc
    self._CheckConflictRegister(desc, desc.full_name, desc.file.name)
    self._enum_descriptors[enum_name] = desc
    if top_level:
        for value in values:
            full_name = _NormalizeFullyQualifiedName('.'.join((package, value.name)))
            self._CheckConflictRegister(value, full_name, file_name)
            self._top_enum_values[full_name] = value
    return desc