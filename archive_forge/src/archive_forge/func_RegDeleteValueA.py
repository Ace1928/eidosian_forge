from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteValueA(hKeySrc, lpValueName=None):
    _RegDeleteValueA = windll.advapi32.RegDeleteValueA
    _RegDeleteValueA.argtypes = [HKEY, LPSTR]
    _RegDeleteValueA.restype = LONG
    _RegDeleteValueA.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteValueA(hKeySrc, lpValueName)