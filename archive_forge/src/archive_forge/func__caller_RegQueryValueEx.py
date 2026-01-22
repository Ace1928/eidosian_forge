from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def _caller_RegQueryValueEx(ansi):
    if ansi:
        _RegQueryValueEx = windll.advapi32.RegQueryValueExA
        _RegQueryValueEx.argtypes = [HKEY, LPSTR, LPVOID, PDWORD, LPVOID, PDWORD]
    else:
        _RegQueryValueEx = windll.advapi32.RegQueryValueExW
        _RegQueryValueEx.argtypes = [HKEY, LPWSTR, LPVOID, PDWORD, LPVOID, PDWORD]
    _RegQueryValueEx.restype = LONG
    _RegQueryValueEx.errcheck = RaiseIfNotErrorSuccess
    return _RegQueryValueEx