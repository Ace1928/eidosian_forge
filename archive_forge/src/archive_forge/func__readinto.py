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
def _readinto(self, buf, read1):
    """Read data into *buf* with at most one system call."""
    if not isinstance(buf, memoryview):
        buf = memoryview(buf)
    if buf.nbytes == 0:
        return 0
    buf = buf.cast('B')
    written = 0
    with self._read_lock:
        while written < len(buf):
            avail = min(len(self._read_buf) - self._read_pos, len(buf))
            if avail:
                buf[written:written + avail] = self._read_buf[self._read_pos:self._read_pos + avail]
                self._read_pos += avail
                written += avail
                if written == len(buf):
                    break
            if len(buf) - written > self.buffer_size:
                n = self.raw.readinto(buf[written:])
                if not n:
                    break
                written += n
            elif not (read1 and written):
                if not self._peek_unlocked(1):
                    break
            if read1 and written:
                break
    return written