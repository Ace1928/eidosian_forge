import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _AddExtensionDescriptor(self, extension):
    """Adds a FieldDescriptor describing an extension to the pool.

    Args:
      extension: A FieldDescriptor.

    Raises:
      AssertionError: when another extension with the same number extends the
        same message.
      TypeError: when the specified extension is not a
        descriptor.FieldDescriptor.
    """
    if not (isinstance(extension, descriptor.FieldDescriptor) and extension.is_extension):
        raise TypeError('Expected an extension descriptor.')
    if extension.extension_scope is None:
        self._CheckConflictRegister(extension, extension.full_name, extension.file.name)
        self._toplevel_extensions[extension.full_name] = extension
    try:
        existing_desc = self._extensions_by_number[extension.containing_type][extension.number]
    except KeyError:
        pass
    else:
        if extension is not existing_desc:
            raise AssertionError('Extensions "%s" and "%s" both try to extend message type "%s" with field number %d.' % (extension.full_name, existing_desc.full_name, extension.containing_type.full_name, extension.number))
    self._extensions_by_number[extension.containing_type][extension.number] = extension
    self._extensions_by_name[extension.containing_type][extension.full_name] = extension
    if _IsMessageSetExtension(extension):
        self._extensions_by_name[extension.containing_type][extension.message_type.full_name] = extension
    if hasattr(extension.containing_type, '_concrete_class'):
        python_message._AttachFieldHelpers(extension.containing_type._concrete_class, extension)