from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import upload_stream
def _read_from_buffer(self, amount):
    """Get any buffered data required to complete a read.

    If a backward seek has not happened, the buffer will never contain any
    information needed to complete a read call. Return the empty string in
    these cases.

    If the current position is before the end of the buffer, some of the
    requested bytes will be in the buffer. For example, if our position is 1,
    five bytes are being read, and the buffer contains b'0123', we will return
    b'123'. Two additional bytes will be read from the stream at a later stage.

    Args:
      amount (int): The total number of bytes to be read

    Returns:
      A byte string, the length of which is equal to `amount` if there are
      enough buffered bytes to complete the read, or less than `amount` if there
      are not.
    """
    buffered_data = []
    bytes_remaining = amount
    if self._position < self._buffer_end:
        position_in_buffer = self._buffer_start
        for data in self._buffer:
            if position_in_buffer + len(data) >= self._position:
                offset_from_position = self._position - position_in_buffer
                bytes_to_read_this_block = len(data) - offset_from_position
                read_size = min(bytes_to_read_this_block, bytes_remaining)
                buffered_data.append(data[offset_from_position:offset_from_position + read_size])
                bytes_remaining -= read_size
                self._position += read_size
            position_in_buffer += len(data)
    return b''.join(buffered_data)