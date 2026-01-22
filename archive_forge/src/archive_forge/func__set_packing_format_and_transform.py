from functools import partial
import mmap
import os
import errno
import struct
import secrets
import types
from . import resource_tracker
def _set_packing_format_and_transform(self, position, fmt_as_str, value):
    """Sets the packing format and back transformation code for a
        single value in the list at the specified position."""
    if position >= self._list_len or self._list_len < 0:
        raise IndexError('Requested position out of range.')
    struct.pack_into('8s', self.shm.buf, self._offset_packing_formats + position * 8, fmt_as_str.encode(_encoding))
    transform_code = self._extract_recreation_code(value)
    struct.pack_into('b', self.shm.buf, self._offset_back_transform_codes + position, transform_code)