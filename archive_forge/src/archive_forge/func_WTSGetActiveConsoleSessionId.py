from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
def WTSGetActiveConsoleSessionId():
    _WTSGetActiveConsoleSessionId = windll.kernel32.WTSGetActiveConsoleSessionId
    _WTSGetActiveConsoleSessionId.argtypes = []
    _WTSGetActiveConsoleSessionId.restype = DWORD
    _WTSGetActiveConsoleSessionId.errcheck = RaiseIfZero
    return _WTSGetActiveConsoleSessionId()