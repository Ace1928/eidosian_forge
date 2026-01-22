import sys
from ctypes import *
class LittleEndianStructure(Structure, metaclass=_swapped_struct_meta):
    """Structure with little endian byte order"""
    __slots__ = ()
    _swappedbytes_ = None