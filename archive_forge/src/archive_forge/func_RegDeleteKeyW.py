from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteKeyW(hKeySrc, lpSubKey=None):
    _RegDeleteKeyW = windll.advapi32.RegDeleteKeyW
    _RegDeleteKeyW.argtypes = [HKEY, LPWSTR]
    _RegDeleteKeyW.restype = LONG
    _RegDeleteKeyW.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteKeyW(hKeySrc, lpSubKey)