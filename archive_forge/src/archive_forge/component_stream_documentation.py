from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import upload_stream
Goes to a specific point in the stream.

    Args:
      offset (int): The number of bytes to move.
      whence: Specifies the position offset is added to.
        os.SEEK_END: offset is added to the last byte in the FilePart.
        os.SEEK_CUR: offset is added to the current position.
        os.SEEK_SET: offset is added to the first byte in the FilePart.

    Returns:
      The new relative position in the stream (int).
    