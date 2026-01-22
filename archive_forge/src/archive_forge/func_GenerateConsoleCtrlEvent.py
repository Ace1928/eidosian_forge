import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GenerateConsoleCtrlEvent(dwCtrlEvent, dwProcessGroupId):
    _GenerateConsoleCtrlEvent = windll.kernel32.GenerateConsoleCtrlEvent
    _GenerateConsoleCtrlEvent.argtypes = [DWORD, DWORD]
    _GenerateConsoleCtrlEvent.restype = bool
    _GenerateConsoleCtrlEvent.errcheck = RaiseIfZero
    _GenerateConsoleCtrlEvent(dwCtrlEvent, dwProcessGroupId)