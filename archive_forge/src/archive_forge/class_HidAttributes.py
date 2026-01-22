import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
class HidAttributes(ctypes.Structure):
    _fields_ = [('Size', ctypes.c_ulong), ('VendorID', ctypes.c_ushort), ('ProductID', ctypes.c_ushort), ('VersionNumber', ctypes.c_ushort)]