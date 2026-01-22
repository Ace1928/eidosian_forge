from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class TEB_ACTIVE_FRAME(Structure):
    _fields_ = [('Flags', DWORD), ('Previous', LPVOID), ('Context', LPVOID)]