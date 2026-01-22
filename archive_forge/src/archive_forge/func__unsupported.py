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
def _unsupported(self, name):
    """Internal: raise an OSError exception for unsupported operations."""
    raise UnsupportedOperation('%s.%s() not supported' % (self.__class__.__name__, name))