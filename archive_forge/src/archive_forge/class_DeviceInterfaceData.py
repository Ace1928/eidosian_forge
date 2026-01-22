import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
class DeviceInterfaceData(ctypes.Structure):
    _fields_ = [('cbSize', wintypes.DWORD), ('InterfaceClassGuid', GUID), ('Flags', wintypes.DWORD), ('Reserved', ctypes.POINTER(ctypes.c_ulong))]
    _pack_ = SETUPAPI_PACK