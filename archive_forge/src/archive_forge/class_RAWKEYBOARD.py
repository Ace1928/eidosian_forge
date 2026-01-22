import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class RAWKEYBOARD(Structure):
    _fields_ = [('MakeCode', USHORT), ('Flags', USHORT), ('Reserved', USHORT), ('VKey', USHORT), ('Message', UINT), ('ExtraInformation', ULONG)]