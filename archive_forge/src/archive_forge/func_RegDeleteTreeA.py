from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteTreeA(hKey, lpSubKey=None):
    _RegDeleteTreeA = windll.advapi32.RegDeleteTreeA
    _RegDeleteTreeA.argtypes = [HKEY, LPWSTR]
    _RegDeleteTreeA.restype = LONG
    _RegDeleteTreeA.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteTreeA(hKey, lpSubKey)