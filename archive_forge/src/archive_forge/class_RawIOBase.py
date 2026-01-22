import _io
import abc
from _io import (DEFAULT_BUFFER_SIZE, BlockingIOError, UnsupportedOperation,
class RawIOBase(_io._RawIOBase, IOBase):
    __doc__ = _io._RawIOBase.__doc__