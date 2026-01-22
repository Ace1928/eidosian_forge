from __future__ import print_function, unicode_literals
import typing
import array
import io
from io import SEEK_CUR, SEEK_SET
from .mode import Mode
class RawWrapper(io.RawIOBase):
    """Convert a Python 2 style file-like object in to a IO object."""

    def __init__(self, f, mode=None, name=None):
        self._f = f
        self.mode = mode or getattr(f, 'mode', None)
        self.name = name
        super(RawWrapper, self).__init__()

    def close(self):
        if not self.closed:
            super(RawWrapper, self).close()
            self._f.close()

    def fileno(self):
        return self._f.fileno()

    def flush(self):
        return self._f.flush()

    def isatty(self):
        return self._f.isatty()

    def seek(self, offset, whence=SEEK_SET):
        return self._f.seek(offset, whence)

    def readable(self):
        return getattr(self._f, 'readable', lambda: Mode(self.mode).reading)()

    def writable(self):
        return getattr(self._f, 'writable', lambda: Mode(self.mode).writing)()

    def seekable(self):
        try:
            return self._f.seekable()
        except AttributeError:
            try:
                self.seek(0, SEEK_CUR)
            except IOError:
                return False
            else:
                return True

    def tell(self):
        return self._f.tell()

    def truncate(self, size=None):
        return self._f.truncate(size)

    def write(self, data):
        if isinstance(data, array.array):
            count = self._f.write(data.tobytes())
        else:
            count = self._f.write(data)
        return len(data) if count is None else count

    @typing.no_type_check
    def read(self, n=-1):
        if n == -1:
            return self.readall()
        return self._f.read(n)

    def read1(self, n=-1):
        return getattr(self._f, 'read1', self.read)(n)

    @typing.no_type_check
    def readall(self):
        return self._f.read()

    @typing.no_type_check
    def readinto(self, b):
        try:
            return self._f.readinto(b)
        except AttributeError:
            data = self._f.read(len(b))
            bytes_read = len(data)
            b[:bytes_read] = data
            return bytes_read

    @typing.no_type_check
    def readinto1(self, b):
        try:
            return self._f.readinto1(b)
        except AttributeError:
            data = self._f.read1(len(b))
            bytes_read = len(data)
            b[:bytes_read] = data
            return bytes_read

    def readline(self, limit=None):
        return self._f.readline(-1 if limit is None else limit)

    def readlines(self, hint=None):
        return self._f.readlines(-1 if hint is None else hint)

    def writelines(self, lines):
        _lines = (line.tobytes() if isinstance(line, array.array) else line for line in lines)
        return self._f.writelines(typing.cast('Iterable[bytes]', _lines))

    def __iter__(self):
        return iter(self._f)