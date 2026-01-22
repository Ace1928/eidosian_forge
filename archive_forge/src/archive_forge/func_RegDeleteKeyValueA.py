from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteKeyValueA(hKeySrc, lpSubKey=None, lpValueName=None):
    _RegDeleteKeyValueA = windll.advapi32.RegDeleteKeyValueA
    _RegDeleteKeyValueA.argtypes = [HKEY, LPSTR, LPSTR]
    _RegDeleteKeyValueA.restype = LONG
    _RegDeleteKeyValueA.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteKeyValueA(hKeySrc, lpSubKey, lpValueName)