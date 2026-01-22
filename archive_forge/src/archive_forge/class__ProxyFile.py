import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
class _ProxyFile:
    """A read-only wrapper of a file."""

    def __init__(self, f, pos=None):
        """Initialize a _ProxyFile."""
        self._file = f
        if pos is None:
            self._pos = f.tell()
        else:
            self._pos = pos

    def read(self, size=None):
        """Read bytes."""
        return self._read(size, self._file.read)

    def read1(self, size=None):
        """Read bytes."""
        return self._read(size, self._file.read1)

    def readline(self, size=None):
        """Read a line."""
        return self._read(size, self._file.readline)

    def readlines(self, sizehint=None):
        """Read multiple lines."""
        result = []
        for line in self:
            result.append(line)
            if sizehint is not None:
                sizehint -= len(line)
                if sizehint <= 0:
                    break
        return result

    def __iter__(self):
        """Iterate over lines."""
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def tell(self):
        """Return the position."""
        return self._pos

    def seek(self, offset, whence=0):
        """Change position."""
        if whence == 1:
            self._file.seek(self._pos)
        self._file.seek(offset, whence)
        self._pos = self._file.tell()

    def close(self):
        """Close the file."""
        if hasattr(self, '_file'):
            try:
                if hasattr(self._file, 'close'):
                    self._file.close()
            finally:
                del self._file

    def _read(self, size, read_method):
        """Read size bytes using read_method."""
        if size is None:
            size = -1
        self._file.seek(self._pos)
        result = read_method(size)
        self._pos = self._file.tell()
        return result

    def __enter__(self):
        """Context management protocol support."""
        return self

    def __exit__(self, *exc):
        self.close()

    def readable(self):
        return self._file.readable()

    def writable(self):
        return self._file.writable()

    def seekable(self):
        return self._file.seekable()

    def flush(self):
        return self._file.flush()

    @property
    def closed(self):
        if not hasattr(self, '_file'):
            return True
        if not hasattr(self._file, 'closed'):
            return False
        return self._file.closed
    __class_getitem__ = classmethod(GenericAlias)