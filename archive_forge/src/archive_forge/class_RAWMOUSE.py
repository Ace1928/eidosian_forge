import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class RAWMOUSE(Structure):
    _anonymous_ = ('u',)
    _fields_ = [('usFlags', USHORT), ('u', _U), ('ulRawButtons', ULONG), ('lLastX', LONG), ('lLastY', LONG), ('ulExtraInformation', ULONG)]