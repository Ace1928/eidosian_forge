import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindAllExtensions(self, message_descriptor):
    """Gets all the known extensions of a given message.

    Extensions have to be registered to this pool by build related
    :func:`Add` or :func:`AddExtensionDescriptor`.

    Args:
      message_descriptor (Descriptor): Descriptor of the extended message.

    Returns:
      list[FieldDescriptor]: Field descriptors describing the extensions.
    """
    if self._descriptor_db and hasattr(self._descriptor_db, 'FindAllExtensionNumbers'):
        full_name = message_descriptor.full_name
        all_numbers = self._descriptor_db.FindAllExtensionNumbers(full_name)
        for number in all_numbers:
            if number in self._extensions_by_number[message_descriptor]:
                continue
            self._TryLoadExtensionFromDB(message_descriptor, number)
    return list(self._extensions_by_number[message_descriptor].values())