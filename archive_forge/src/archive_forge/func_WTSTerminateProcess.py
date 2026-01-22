from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
def WTSTerminateProcess(hServer, ProcessId, ExitCode):
    _WTSTerminateProcess = windll.wtsapi32.WTSTerminateProcess
    _WTSTerminateProcess.argtypes = [HANDLE, DWORD, DWORD]
    _WTSTerminateProcess.restype = bool
    _WTSTerminateProcess.errcheck = RaiseIfZero
    _WTSTerminateProcess(hServer, ProcessId, ExitCode)