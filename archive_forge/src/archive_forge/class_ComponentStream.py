from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import upload_stream
class ComponentStream(upload_stream.UploadStream):
    """Implements a subset of the io.IOBase API exposing part of a stream.

  This class behaves as a contiguous subset of the underlying stream.

  This is helpful for composite uploads since even when total_size is specified,
  apitools.transfer.Upload looks at the size of the source file to ensure
  that all bytes have been uploaded.
  """

    def __init__(self, stream, offset, length, digesters=None, progress_callback=None):
        """Initializes a ComponentStream instance.

    Args:
      stream (io.IOBase): See super class.
      offset (int|None): The position (in bytes) in the wrapped stream that
        corresponds to the first byte of the ComponentStream.
      length (int|None): The total number of bytes readable from the
        ComponentStream.
      digesters (dict[util.HashAlgorithm, hashlib hash object]|None): See super
        class.
      progress_callback (func[int]|None): See super class.
    """
        super(__class__, self).__init__(stream=stream, length=length, digesters=digesters, progress_callback=progress_callback)
        self._start_byte = offset
        self._end_byte = self._start_byte + self._length
        self._stream.seek(self._start_byte)

    def tell(self):
        """Returns the current position relative to the part's start byte."""
        return super(__class__, self).tell() - self._start_byte

    def read(self, size=-1):
        """Reads `size` bytes from a stream, or all bytes when `size` < 0."""
        if size < 0:
            size = self._length
        size = min(size, self._end_byte - super(__class__, self).tell())
        return super(__class__, self).read(max(0, size))

    def seek(self, offset, whence=os.SEEK_SET):
        """Goes to a specific point in the stream.

    Args:
      offset (int): The number of bytes to move.
      whence: Specifies the position offset is added to.
        os.SEEK_END: offset is added to the last byte in the FilePart.
        os.SEEK_CUR: offset is added to the current position.
        os.SEEK_SET: offset is added to the first byte in the FilePart.

    Returns:
      The new relative position in the stream (int).
    """
        if whence == os.SEEK_END:
            new_absolute_index = offset + self._end_byte
        elif whence == os.SEEK_CUR:
            new_absolute_index = super(__class__, self).tell() + offset
        else:
            new_absolute_index = offset + self._start_byte
        return super(__class__, self).seek(new_absolute_index) - self._start_byte