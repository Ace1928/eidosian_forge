import _io
import abc
from _io import (DEFAULT_BUFFER_SIZE, BlockingIOError, UnsupportedOperation,
class IOBase(_io._IOBase, metaclass=abc.ABCMeta):
    __doc__ = _io._IOBase.__doc__