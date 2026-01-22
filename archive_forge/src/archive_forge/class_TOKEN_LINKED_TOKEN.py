from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class TOKEN_LINKED_TOKEN(Structure):
    _fields_ = [('LinkedToken', HANDLE)]