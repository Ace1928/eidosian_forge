import ctypes
from os_win.utils.winapi import wintypes
class RESIZE_VIRTUAL_DISK_PARAMETERS(ctypes.Structure):
    _fields_ = [('Version', wintypes.DWORD), ('Version1', _RESIZE_VIRTUAL_DISK_PARAMETERS_V1)]