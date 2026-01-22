import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def ContinueDebugEvent(dwProcessId, dwThreadId, dwContinueStatus=DBG_EXCEPTION_NOT_HANDLED):
    _ContinueDebugEvent = windll.kernel32.ContinueDebugEvent
    _ContinueDebugEvent.argtypes = [DWORD, DWORD, DWORD]
    _ContinueDebugEvent.restype = bool
    _ContinueDebugEvent.errcheck = RaiseIfZero
    _ContinueDebugEvent(dwProcessId, dwThreadId, dwContinueStatus)