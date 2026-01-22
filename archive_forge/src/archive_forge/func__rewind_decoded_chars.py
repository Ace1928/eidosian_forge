import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _rewind_decoded_chars(self, n):
    """Rewind the _decoded_chars buffer."""
    if self._decoded_chars_used < n:
        raise AssertionError('rewind decoded_chars out of bounds')
    self._decoded_chars_used -= n