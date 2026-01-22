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
def _read_unlocked(self, n=None):
    nodata_val = b''
    empty_values = (b'', None)
    buf = self._read_buf
    pos = self._read_pos
    if n is None or n == -1:
        self._reset_read_buf()
        if hasattr(self.raw, 'readall'):
            chunk = self.raw.readall()
            if chunk is None:
                return buf[pos:] or None
            else:
                return buf[pos:] + chunk
        chunks = [buf[pos:]]
        current_size = 0
        while True:
            chunk = self.raw.read()
            if chunk in empty_values:
                nodata_val = chunk
                break
            current_size += len(chunk)
            chunks.append(chunk)
        return b''.join(chunks) or nodata_val
    avail = len(buf) - pos
    if n <= avail:
        self._read_pos += n
        return buf[pos:pos + n]
    chunks = [buf[pos:]]
    wanted = max(self.buffer_size, n)
    while avail < n:
        chunk = self.raw.read(wanted)
        if chunk in empty_values:
            nodata_val = chunk
            break
        avail += len(chunk)
        chunks.append(chunk)
    n = min(n, avail)
    out = b''.join(chunks)
    self._read_buf = out[n:]
    self._read_pos = 0
    return out[:n] if out else nodata_val