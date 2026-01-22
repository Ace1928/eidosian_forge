from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
def WTSEnumerateProcessesA(hServer=WTS_CURRENT_SERVER_HANDLE):
    _WTSEnumerateProcessesA = windll.wtsapi32.WTSEnumerateProcessesA
    _WTSEnumerateProcessesA.argtypes = [HANDLE, DWORD, DWORD, POINTER(PWTS_PROCESS_INFOA), PDWORD]
    _WTSEnumerateProcessesA.restype = bool
    _WTSEnumerateProcessesA.errcheck = RaiseIfZero
    pProcessInfo = PWTS_PROCESS_INFOA()
    Count = DWORD(0)
    _WTSEnumerateProcessesA(hServer, 0, 1, byref(pProcessInfo), byref(Count))
    return (pProcessInfo, Count.value)