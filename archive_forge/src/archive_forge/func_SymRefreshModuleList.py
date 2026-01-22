from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymRefreshModuleList(hProcess):
    _SymRefreshModuleList = windll.dbghelp.SymRefreshModuleList
    _SymRefreshModuleList.argtypes = [HANDLE]
    _SymRefreshModuleList.restype = bool
    _SymRefreshModuleList.errcheck = RaiseIfZero
    _SymRefreshModuleList(hProcess)