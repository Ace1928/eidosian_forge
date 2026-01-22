import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def AddExtensionForNested(message_type):
    for nested in message_type.nested_types:
        AddExtensionForNested(nested)
    for extension in message_type.extensions:
        self._AddExtensionDescriptor(extension)