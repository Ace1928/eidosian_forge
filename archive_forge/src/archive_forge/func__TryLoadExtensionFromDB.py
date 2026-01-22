import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _TryLoadExtensionFromDB(self, message_descriptor, number):
    """Try to Load extensions from descriptor db.

    Args:
      message_descriptor: descriptor of the extended message.
      number: the extension number that needs to be loaded.
    """
    if not self._descriptor_db:
        return
    if not hasattr(self._descriptor_db, 'FindFileContainingExtension'):
        return
    full_name = message_descriptor.full_name
    file_proto = self._descriptor_db.FindFileContainingExtension(full_name, number)
    if file_proto is None:
        return
    try:
        self._ConvertFileProtoToFileDescriptor(file_proto)
    except:
        warn_msg = 'Unable to load proto file %s for extension number %d.' % (file_proto.name, number)
        warnings.warn(warn_msg, RuntimeWarning)