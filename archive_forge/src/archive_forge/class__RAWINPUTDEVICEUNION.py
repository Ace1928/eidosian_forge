import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class _RAWINPUTDEVICEUNION(Union):
    _fields_ = [('mouse', RAWMOUSE), ('keyboard', RAWKEYBOARD), ('hid', RAWHID)]