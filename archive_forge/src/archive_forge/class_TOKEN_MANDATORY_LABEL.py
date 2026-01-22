from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_MANDATORY_LABEL(Structure):
    _fields_ = [('Label', SID_AND_ATTRIBUTES)]