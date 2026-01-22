import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindExtensionByNumber(self, message_descriptor, number):
    """Gets the extension of the specified message with the specified number.

    Extensions have to be registered to this pool by calling :func:`Add` or
    :func:`AddExtensionDescriptor`.

    Args:
      message_descriptor (Descriptor): descriptor of the extended message.
      number (int): Number of the extension field.

    Returns:
      FieldDescriptor: The descriptor for the extension.

    Raises:
      KeyError: when no extension with the given number is known for the
        specified message.
    """
    try:
        return self._extensions_by_number[message_descriptor][number]
    except KeyError:
        self._TryLoadExtensionFromDB(message_descriptor, number)
        return self._extensions_by_number[message_descriptor][number]