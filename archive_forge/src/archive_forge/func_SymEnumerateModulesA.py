from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymEnumerateModulesA(hProcess, EnumModulesCallback, UserContext=None):
    _SymEnumerateModules = windll.dbghelp.SymEnumerateModules
    _SymEnumerateModules.argtypes = [HANDLE, PSYM_ENUMMODULES_CALLBACK, PVOID]
    _SymEnumerateModules.restype = bool
    _SymEnumerateModules.errcheck = RaiseIfZero
    EnumModulesCallback = PSYM_ENUMMODULES_CALLBACK(EnumModulesCallback)
    if UserContext:
        UserContext = ctypes.pointer(UserContext)
    else:
        UserContext = LPVOID(NULL)
    _SymEnumerateModules(hProcess, EnumModulesCallback, UserContext)