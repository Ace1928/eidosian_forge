from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegFlushKey(hKey):
    _RegFlushKey = windll.advapi32.RegFlushKey
    _RegFlushKey.argtypes = [HKEY]
    _RegFlushKey.restype = LONG
    _RegFlushKey.errcheck = RaiseIfNotErrorSuccess
    _RegFlushKey(hKey)