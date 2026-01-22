import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetExitCodeThread(hThread):
    _GetExitCodeThread = windll.kernel32.GetExitCodeThread
    _GetExitCodeThread.argtypes = [HANDLE]
    _GetExitCodeThread.restype = bool
    _GetExitCodeThread.errcheck = RaiseIfZero
    lpExitCode = DWORD(0)
    _GetExitCodeThread(hThread, byref(lpExitCode))
    return lpExitCode.value