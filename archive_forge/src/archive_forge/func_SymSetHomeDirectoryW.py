from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymSetHomeDirectoryW(hProcess, dir=None):
    _SymSetHomeDirectoryW = windll.dbghelp.SymSetHomeDirectoryW
    _SymSetHomeDirectoryW.argtypes = [HANDLE, LPWSTR]
    _SymSetHomeDirectoryW.restype = LPWSTR
    _SymSetHomeDirectoryW.errcheck = RaiseIfZero
    if not dir:
        dir = None
    _SymSetHomeDirectoryW(hProcess, dir)
    return dir