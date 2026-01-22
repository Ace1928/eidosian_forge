from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymUnloadModule64(hProcess, BaseOfDll):
    _SymUnloadModule64 = windll.dbghelp.SymUnloadModule64
    _SymUnloadModule64.argtypes = [HANDLE, DWORD64]
    _SymUnloadModule64.restype = bool
    _SymUnloadModule64.errcheck = RaiseIfZero
    _SymUnloadModule64(hProcess, BaseOfDll)