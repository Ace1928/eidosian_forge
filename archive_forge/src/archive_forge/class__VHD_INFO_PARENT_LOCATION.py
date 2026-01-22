import ctypes
from os_win.utils.winapi import wintypes
class _VHD_INFO_PARENT_LOCATION(ctypes.Structure):
    _fields_ = [('ParentResolved', wintypes.BOOL), ('ParentPath', wintypes.WCHAR * 512)]