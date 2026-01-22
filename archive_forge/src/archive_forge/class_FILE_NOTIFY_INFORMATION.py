from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
class FILE_NOTIFY_INFORMATION(ctypes.Structure):
    _fields_ = [('NextEntryOffset', ctypes.wintypes.DWORD), ('Action', ctypes.wintypes.DWORD), ('FileNameLength', ctypes.wintypes.DWORD), ('FileName', ctypes.c_char * 1)]