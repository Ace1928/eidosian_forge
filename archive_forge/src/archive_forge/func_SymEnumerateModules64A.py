from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateModules64A(hProcess, EnumModulesCallback, UserContext=None):
    _SymEnumerateModules64 = windll.dbghelp.SymEnumerateModules64
    _SymEnumerateModules64.argtypes = [HANDLE, PSYM_ENUMMODULES_CALLBACK64, PVOID]
    _SymEnumerateModules64.restype = bool
    _SymEnumerateModules64.errcheck = RaiseIfZero
    EnumModulesCallback = PSYM_ENUMMODULES_CALLBACK64(EnumModulesCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateModules64(hProcess, EnumModulesCallback, UserContext)