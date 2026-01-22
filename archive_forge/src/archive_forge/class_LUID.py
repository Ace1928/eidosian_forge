from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class LUID(Structure):
    _fields_ = [('LowPart', DWORD), ('HighPart', LONG)]