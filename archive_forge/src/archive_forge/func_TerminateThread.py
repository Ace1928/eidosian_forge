import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def TerminateThread(hThread, dwExitCode=0):
    _TerminateThread = windll.kernel32.TerminateThread
    _TerminateThread.argtypes = [HANDLE, DWORD]
    _TerminateThread.restype = bool
    _TerminateThread.errcheck = RaiseIfZero
    _TerminateThread(hThread, dwExitCode)