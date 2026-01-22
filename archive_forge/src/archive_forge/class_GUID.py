import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
class GUID(ctypes.Structure):
    _fields_ = [('Data1', ctypes.c_ulong), ('Data2', ctypes.c_ushort), ('Data3', ctypes.c_ushort), ('Data4', ctypes.c_ubyte * 8)]