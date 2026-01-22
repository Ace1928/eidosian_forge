from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class IMAGEHLP_SYMBOL64(Structure):
    _fields_ = [('SizeOfStruct', DWORD), ('Address', DWORD64), ('Size', DWORD), ('Flags', DWORD), ('MaxNameLength', DWORD), ('Name', CHAR * (MAX_SYM_NAME + 1))]