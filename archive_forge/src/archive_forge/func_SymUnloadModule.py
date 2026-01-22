from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymUnloadModule(hProcess, BaseOfDll):
    _SymUnloadModule = windll.dbghelp.SymUnloadModule
    _SymUnloadModule.argtypes = [HANDLE, DWORD]
    _SymUnloadModule.restype = bool
    _SymUnloadModule.errcheck = RaiseIfZero
    _SymUnloadModule(hProcess, BaseOfDll)