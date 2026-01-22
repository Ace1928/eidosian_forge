from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_ORIGIN(Structure):
    _fields_ = [('OriginatingLogonSession', LUID)]