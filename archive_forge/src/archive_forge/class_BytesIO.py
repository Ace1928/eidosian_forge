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
class BytesIO(BufferedIOBase):
    """Buffered I/O implementation using an in-memory bytes buffer."""
    _buffer = None

    def __init__(self, initial_bytes=None):
        buf = bytearray()
        if initial_bytes is not None:
            buf += initial_bytes
        self._buffer = buf
        self._pos = 0

    def __getstate__(self):
        if self.closed:
            raise ValueError('__getstate__ on closed file')
        return self.__dict__.copy()

    def getvalue(self):
        """Return the bytes value (contents) of the buffer
        """
        if self.closed:
            raise ValueError('getvalue on closed file')
        return bytes(self._buffer)

    def getbuffer(self):
        """Return a readable and writable view of the buffer.
        """
        if self.closed:
            raise ValueError('getbuffer on closed file')
        return memoryview(self._buffer)

    def close(self):
        if self._buffer is not None:
            self._buffer.clear()
        super().close()

    def read(self, size=-1):
        if self.closed:
            raise ValueError('read from closed file')
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f'{size!r} is not an integer')
            else:
                size = size_index()
        if size < 0:
            size = len(self._buffer)
        if len(self._buffer) <= self._pos:
            return b''
        newpos = min(len(self._buffer), self._pos + size)
        b = self._buffer[self._pos:newpos]
        self._pos = newpos
        return bytes(b)

    def read1(self, size=-1):
        """This is the same as read.
        """
        return self.read(size)

    def write(self, b):
        if self.closed:
            raise ValueError('write to closed file')
        if isinstance(b, str):
            raise TypeError("can't write str to binary stream")
        with memoryview(b) as view:
            n = view.nbytes
        if n == 0:
            return 0
        pos = self._pos
        if pos > len(self._buffer):
            padding = b'\x00' * (pos - len(self._buffer))
            self._buffer += padding
        self._buffer[pos:pos + n] = b
        self._pos += n
        return n

    def seek(self, pos, whence=0):
        if self.closed:
            raise ValueError('seek on closed file')
        try:
            pos_index = pos.__index__
        except AttributeError:
            raise TypeError(f'{pos!r} is not an integer')
        else:
            pos = pos_index()
        if whence == 0:
            if pos < 0:
                raise ValueError('negative seek position %r' % (pos,))
            self._pos = pos
        elif whence == 1:
            self._pos = max(0, self._pos + pos)
        elif whence == 2:
            self._pos = max(0, len(self._buffer) + pos)
        else:
            raise ValueError('unsupported whence value')
        return self._pos

    def tell(self):
        if self.closed:
            raise ValueError('tell on closed file')
        return self._pos

    def truncate(self, pos=None):
        if self.closed:
            raise ValueError('truncate on closed file')
        if pos is None:
            pos = self._pos
        else:
            try:
                pos_index = pos.__index__
            except AttributeError:
                raise TypeError(f'{pos!r} is not an integer')
            else:
                pos = pos_index()
            if pos < 0:
                raise ValueError('negative truncate position %r' % (pos,))
        del self._buffer[pos:]
        return pos

    def readable(self):
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return True

    def writable(self):
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return True

    def seekable(self):
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        return True