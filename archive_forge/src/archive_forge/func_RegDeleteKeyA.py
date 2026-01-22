from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteKeyA(hKeySrc, lpSubKey=None):
    _RegDeleteKeyA = windll.advapi32.RegDeleteKeyA
    _RegDeleteKeyA.argtypes = [HKEY, LPSTR]
    _RegDeleteKeyA.restype = LONG
    _RegDeleteKeyA.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteKeyA(hKeySrc, lpSubKey)