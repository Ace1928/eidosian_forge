import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def AddSerializedFile(self, serialized_file_desc_proto):
    """Adds the FileDescriptorProto and its types to this pool.

    Args:
      serialized_file_desc_proto (bytes): A bytes string, serialization of the
        :class:`FileDescriptorProto` to add.

    Returns:
      FileDescriptor: Descriptor for the added file.
    """
    from google.protobuf import descriptor_pb2
    file_desc_proto = descriptor_pb2.FileDescriptorProto.FromString(serialized_file_desc_proto)
    file_desc = self._ConvertFileProtoToFileDescriptor(file_desc_proto)
    file_desc.serialized_pb = serialized_file_desc_proto
    return file_desc