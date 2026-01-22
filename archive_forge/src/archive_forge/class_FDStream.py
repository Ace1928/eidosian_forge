from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class FDStream:
    """A simple wrapper providing the most basic functions on a file descriptor
    with the fileobject interface. Cannot use os.fdopen as the resulting stream
    takes ownership"""
    __slots__ = ('_fd', '_pos')

    def __init__(self, fd):
        self._fd = fd
        self._pos = 0

    def write(self, data):
        self._pos += len(data)
        os.write(self._fd, data)

    def read(self, count=0):
        if count == 0:
            count = os.path.getsize(self._filepath)
        bytes = os.read(self._fd, count)
        self._pos += len(bytes)
        return bytes

    def fileno(self):
        return self._fd

    def tell(self):
        return self._pos

    def close(self):
        close(self._fd)