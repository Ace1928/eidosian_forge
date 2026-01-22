from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
def ProcessIdToSessionId(dwProcessId):
    _ProcessIdToSessionId = windll.kernel32.ProcessIdToSessionId
    _ProcessIdToSessionId.argtypes = [DWORD, PDWORD]
    _ProcessIdToSessionId.restype = bool
    _ProcessIdToSessionId.errcheck = RaiseIfZero
    dwSessionId = DWORD(0)
    _ProcessIdToSessionId(dwProcessId, byref(dwSessionId))
    return dwSessionId.value