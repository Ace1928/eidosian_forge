from cloudsdk.google.protobuf.internal import api_implementation
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
Gets all the messages from a specified file.

    This will find and resolve dependencies, failing if the descriptor
    pool cannot satisfy them.

    Args:
      files: The file names to extract messages from.

    Returns:
      A dictionary mapping proto names to the message classes. This will include
      any dependent messages as well as any messages defined in the same file as
      a specified message.
    