from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteKeyValueW(hKeySrc, lpSubKey=None, lpValueName=None):
    _RegDeleteKeyValueW = windll.advapi32.RegDeleteKeyValueW
    _RegDeleteKeyValueW.argtypes = [HKEY, LPWSTR, LPWSTR]
    _RegDeleteKeyValueW.restype = LONG
    _RegDeleteKeyValueW.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteKeyValueW(hKeySrc, lpSubKey, lpValueName)