import sys
from ctypes import *
class LittleEndianUnion(Union, metaclass=_swapped_union_meta):
    """Union with little endian byte order"""
    __slots__ = ()
    _swappedbytes_ = None