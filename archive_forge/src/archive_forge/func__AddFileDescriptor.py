import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _AddFileDescriptor(self, file_desc):
    """Adds a FileDescriptor to the pool, non-recursively.

    If the FileDescriptor contains messages or enums, the caller must explicitly
    register them.

    Args:
      file_desc: A FileDescriptor.
    """
    if not isinstance(file_desc, descriptor.FileDescriptor):
        raise TypeError('Expected instance of descriptor.FileDescriptor.')
    self._file_descriptors[file_desc.name] = file_desc