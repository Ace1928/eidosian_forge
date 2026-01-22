import _io
import abc
from _io import (DEFAULT_BUFFER_SIZE, BlockingIOError, UnsupportedOperation,
class BufferedIOBase(_io._BufferedIOBase, IOBase):
    __doc__ = _io._BufferedIOBase.__doc__