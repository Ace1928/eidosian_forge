import sys
from ctypes import *
class BigEndianStructure(Structure, metaclass=_swapped_struct_meta):
    """Structure with big endian byte order"""
    __slots__ = ()
    _swappedbytes_ = None