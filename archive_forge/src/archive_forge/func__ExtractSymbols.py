import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _ExtractSymbols(self, descriptors):
    """Pulls out all the symbols from descriptor protos.

    Args:
      descriptors: The messages to extract descriptors from.
    Yields:
      A two element tuple of the type name and descriptor object.
    """
    for desc in descriptors:
        yield (_PrefixWithDot(desc.full_name), desc)
        for symbol in self._ExtractSymbols(desc.nested_types):
            yield symbol
        for enum in desc.enum_types:
            yield (_PrefixWithDot(enum.full_name), enum)