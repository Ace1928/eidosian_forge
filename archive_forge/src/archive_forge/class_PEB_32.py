from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class PEB_32(Structure):
    _pack_ = 8
    if os == 'Windows NT':
        _pack_ = _PEB_NT._pack_
        _fields_ = _PEB_NT._fields_
    elif os == 'Windows 2000':
        _pack_ = _PEB_2000._pack_
        _fields_ = _PEB_2000._fields_
    elif os.startswith('Windows XP'):
        _fields_ = _PEB_XP._fields_
    elif os.startswith('Windows 2003 R2'):
        _fields_ = _PEB_2003_R2._fields_
    elif os.startswith('Windows 2003'):
        _fields_ = _PEB_2003._fields_
    elif os.startswith('Windows 2008 R2'):
        _fields_ = _PEB_2008_R2._fields_
    elif os.startswith('Windows 2008'):
        _fields_ = _PEB_2008._fields_
    elif os.startswith('Windows Vista'):
        _fields_ = _PEB_Vista._fields_
    else:
        _fields_ = _PEB_W7._fields_