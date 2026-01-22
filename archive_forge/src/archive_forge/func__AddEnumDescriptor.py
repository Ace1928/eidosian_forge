import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _AddEnumDescriptor(self, enum_desc):
    """Adds an EnumDescriptor to the pool.

    This method also registers the FileDescriptor associated with the enum.

    Args:
      enum_desc: An EnumDescriptor.
    """
    if not isinstance(enum_desc, descriptor.EnumDescriptor):
        raise TypeError('Expected instance of descriptor.EnumDescriptor.')
    file_name = enum_desc.file.name
    self._CheckConflictRegister(enum_desc, enum_desc.full_name, file_name)
    self._enum_descriptors[enum_desc.full_name] = enum_desc
    if enum_desc.file.package:
        top_level = enum_desc.full_name.count('.') - enum_desc.file.package.count('.') == 1
    else:
        top_level = enum_desc.full_name.count('.') == 0
    if top_level:
        file_name = enum_desc.file.name
        package = enum_desc.file.package
        for enum_value in enum_desc.values:
            full_name = _NormalizeFullyQualifiedName('.'.join((package, enum_value.name)))
            self._CheckConflictRegister(enum_value, full_name, file_name)
            self._top_enum_values[full_name] = enum_value
    self._AddFileDescriptor(enum_desc.file)