from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class TEB(Structure):
    _pack_ = 8
    if os == 'Windows NT':
        _pack_ = _TEB_NT._pack_
        _fields_ = _TEB_NT._fields_
    elif os == 'Windows 2000':
        _pack_ = _TEB_2000._pack_
        _fields_ = _TEB_2000._fields_
    elif os == 'Windows XP':
        _fields_ = _TEB_XP._fields_
    elif os == 'Windows XP (64 bits)':
        _fields_ = _TEB_XP_64._fields_
    elif os == 'Windows 2003':
        _fields_ = _TEB_2003._fields_
    elif os == 'Windows 2003 (64 bits)':
        _fields_ = _TEB_2003_64._fields_
    elif os == 'Windows 2008':
        _fields_ = _TEB_2008._fields_
    elif os == 'Windows 2008 (64 bits)':
        _fields_ = _TEB_2008_64._fields_
    elif os == 'Windows 2003 R2':
        _fields_ = _TEB_2003_R2._fields_
    elif os == 'Windows 2003 R2 (64 bits)':
        _fields_ = _TEB_2003_R2_64._fields_
    elif os == 'Windows 2008 R2':
        _fields_ = _TEB_2008_R2._fields_
    elif os == 'Windows 2008 R2 (64 bits)':
        _fields_ = _TEB_2008_R2_64._fields_
    elif os == 'Windows Vista':
        _fields_ = _TEB_Vista._fields_
    elif os == 'Windows Vista (64 bits)':
        _fields_ = _TEB_Vista_64._fields_
    elif os == 'Windows 7':
        _fields_ = _TEB_W7._fields_
    elif os == 'Windows 7 (64 bits)':
        _fields_ = _TEB_W7_64._fields_
    elif sizeof(SIZE_T) == sizeof(DWORD):
        _fields_ = _TEB_W7._fields_
    else:
        _fields_ = _TEB_W7_64._fields_