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
def _peek_unlocked(self, n=0):
    want = min(n, self.buffer_size)
    have = len(self._read_buf) - self._read_pos
    if have < want or have <= 0:
        to_read = self.buffer_size - have
        current = self.raw.read(to_read)
        if current:
            self._read_buf = self._read_buf[self._read_pos:] + current
            self._read_pos = 0
    return self._read_buf[self._read_pos:]