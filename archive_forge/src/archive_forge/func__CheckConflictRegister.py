import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _CheckConflictRegister(self, desc, desc_name, file_name):
    """Check if the descriptor name conflicts with another of the same name.

    Args:
      desc: Descriptor of a message, enum, service, extension or enum value.
      desc_name (str): the full name of desc.
      file_name (str): The file name of descriptor.
    """
    for register, descriptor_type in [(self._descriptors, descriptor.Descriptor), (self._enum_descriptors, descriptor.EnumDescriptor), (self._service_descriptors, descriptor.ServiceDescriptor), (self._toplevel_extensions, descriptor.FieldDescriptor), (self._top_enum_values, descriptor.EnumValueDescriptor)]:
        if desc_name in register:
            old_desc = register[desc_name]
            if isinstance(old_desc, descriptor.EnumValueDescriptor):
                old_file = old_desc.type.file.name
            else:
                old_file = old_desc.file.name
            if not isinstance(desc, descriptor_type) or old_file != file_name:
                error_msg = 'Conflict register for file "' + file_name + '": ' + desc_name + ' is already defined in file "' + old_file + '". Please fix the conflict by adding package name on the proto file, or use different name for the duplication.'
                if isinstance(desc, descriptor.EnumValueDescriptor):
                    error_msg += '\nNote: enum values appear as siblings of the enum type instead of children of it.'
                raise TypeError(error_msg)
            return