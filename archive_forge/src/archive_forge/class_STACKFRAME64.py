from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class STACKFRAME64(Structure):
    _fields_ = [('AddrPC', ADDRESS64), ('AddrReturn', ADDRESS64), ('AddrFrame', ADDRESS64), ('AddrStack', ADDRESS64), ('AddrBStore', ADDRESS64), ('FuncTableEntry', PVOID), ('Params', DWORD64 * 4), ('Far', BOOL), ('Virtual', BOOL), ('Reserved', DWORD64 * 3), ('KdHelp', KDHELP64)]