from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class LUID_AND_ATTRIBUTES(Structure):
    _fields_ = [('Luid', LUID), ('Attributes', DWORD)]