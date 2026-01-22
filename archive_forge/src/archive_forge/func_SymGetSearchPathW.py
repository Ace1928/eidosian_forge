from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymGetSearchPathW(hProcess):
    _SymGetSearchPathW = windll.dbghelp.SymGetSearchPathW
    _SymGetSearchPathW.argtypes = [HANDLE, LPWSTR, DWORD]
    _SymGetSearchPathW.restype = bool
    _SymGetSearchPathW.errcheck = RaiseIfZero
    SearchPathLength = MAX_PATH
    SearchPath = ctypes.create_unicode_buffer(u'', SearchPathLength)
    _SymGetSearchPathW(hProcess, SearchPath, SearchPathLength)
    return SearchPath.value