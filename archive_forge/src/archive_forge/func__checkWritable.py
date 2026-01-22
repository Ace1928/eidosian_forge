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
def _checkWritable(self, msg=None):
    if not self._writable:
        raise UnsupportedOperation('File not open for writing')