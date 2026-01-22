import ctypes
from os_win.utils.winapi import wintypes
class _RESIZE_VIRTUAL_DISK_PARAMETERS_V1(ctypes.Structure):
    _fields_ = [('NewSize', wintypes.ULONGLONG)]