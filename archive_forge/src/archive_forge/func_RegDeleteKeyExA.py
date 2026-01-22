from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteKeyExA(hKeySrc, lpSubKey=None, samDesired=KEY_WOW64_32KEY):
    _RegDeleteKeyExA = windll.advapi32.RegDeleteKeyExA
    _RegDeleteKeyExA.argtypes = [HKEY, LPSTR, REGSAM, DWORD]
    _RegDeleteKeyExA.restype = LONG
    _RegDeleteKeyExA.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteKeyExA(hKeySrc, lpSubKey, samDesired, 0)