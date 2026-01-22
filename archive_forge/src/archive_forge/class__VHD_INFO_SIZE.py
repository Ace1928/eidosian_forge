import ctypes
from os_win.utils.winapi import wintypes
class _VHD_INFO_SIZE(ctypes.Structure):
    _fields_ = [('VirtualSize', wintypes.ULARGE_INTEGER), ('PhysicalSize', wintypes.ULARGE_INTEGER), ('BlockSize', wintypes.ULONG), ('SectorSize', wintypes.ULONG)]