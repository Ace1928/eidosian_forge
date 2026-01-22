import ctypes
from os_win.utils.winapi import wintypes
class SET_VIRTUAL_DISK_INFO(ctypes.Structure):
    _anonymous_ = ['_setinfo']
    _fields_ = [('Version', wintypes.DWORD), ('_setinfo', _SET_VIRTUAL_DISK_INFO_U)]