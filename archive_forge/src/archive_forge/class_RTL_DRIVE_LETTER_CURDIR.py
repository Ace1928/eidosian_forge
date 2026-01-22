from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class RTL_DRIVE_LETTER_CURDIR(Structure):
    _fields_ = [('Flags', USHORT), ('Length', USHORT), ('TimeStamp', ULONG), ('DosPath', UNICODE_STRING)]