from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class ADDRESS64(Structure):
    _fields_ = [('Offset', DWORD64), ('Segment', WORD), ('Mode', ADDRESS_MODE)]