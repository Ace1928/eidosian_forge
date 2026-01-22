from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateModules64W(hProcess, EnumModulesCallback, UserContext=None):
    _SymEnumerateModules64W = windll.dbghelp.SymEnumerateModules64W
    _SymEnumerateModules64W.argtypes = [HANDLE, PSYM_ENUMMODULES_CALLBACK64W, PVOID]
    _SymEnumerateModules64W.restype = bool
    _SymEnumerateModules64W.errcheck = RaiseIfZero
    EnumModulesCallback = PSYM_ENUMMODULES_CALLBACK64W(EnumModulesCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateModules64W(hProcess, EnumModulesCallback, UserContext)