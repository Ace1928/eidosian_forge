from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
def _get_packing_format(self, position):
    """Gets the packing format for a single value stored in the list."""
    position = position if position >= 0 else position + self._list_len
    if position >= self._list_len or self._list_len < 0:
        raise IndexError('Requested position out of range.')
    v = struct.unpack_from('8s', self.shm.buf, self._offset_packing_formats + position * 8)[0]
    fmt = v.rstrip(b'\x00')
    fmt_as_str = fmt.decode(_encoding)
    return fmt_as_str