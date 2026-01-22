import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def TerminateProcess(hProcess, dwExitCode=0):
    _TerminateProcess = windll.kernel32.TerminateProcess
    _TerminateProcess.argtypes = [HANDLE, DWORD]
    _TerminateProcess.restype = bool
    _TerminateProcess.errcheck = RaiseIfZero
    _TerminateProcess(hProcess, dwExitCode)