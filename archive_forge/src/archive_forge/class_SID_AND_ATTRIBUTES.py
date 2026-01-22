from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class SID_AND_ATTRIBUTES(Structure):
    _fields_ = [('Sid', PSID), ('Attributes', DWORD)]