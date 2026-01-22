import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class VARIANT(Structure):
    _anonymous_ = ['union']
    _fields_ = [('vt', ctypes.c_ushort), ('wReserved1', WORD), ('wReserved2', WORD), ('wReserved3', WORD), ('union', _VarTableVariant)]