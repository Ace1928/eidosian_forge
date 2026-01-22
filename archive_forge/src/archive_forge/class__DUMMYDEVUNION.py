import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class _DUMMYDEVUNION(Union):
    _anonymous_ = ('_dummystruct1', '_dummystruct2')
    _fields_ = [('_dummystruct1', _DUMMYSTRUCTNAME), ('dmPosition', POINTL), ('_dummystruct2', _DUMMYSTRUCTNAME2)]