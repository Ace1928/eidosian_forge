import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DebugActiveProcess(dwProcessId):
    _DebugActiveProcess = windll.kernel32.DebugActiveProcess
    _DebugActiveProcess.argtypes = [DWORD]
    _DebugActiveProcess.restype = bool
    _DebugActiveProcess.errcheck = RaiseIfZero
    _DebugActiveProcess(dwProcessId)