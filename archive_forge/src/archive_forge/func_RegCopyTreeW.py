from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegCopyTreeW(hKeySrc, lpSubKey, hKeyDest):
    _RegCopyTreeW = windll.advapi32.RegCopyTreeW
    _RegCopyTreeW.argtypes = [HKEY, LPWSTR, HKEY]
    _RegCopyTreeW.restype = LONG
    _RegCopyTreeW.errcheck = RaiseIfNotErrorSuccess
    _RegCopyTreeW(hKeySrc, lpSubKey, hKeyDest)