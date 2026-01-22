import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessDEPPolicy(hProcess):
    _GetProcessDEPPolicy = windll.kernel32.GetProcessDEPPolicy
    _GetProcessDEPPolicy.argtypes = [HANDLE, LPDWORD, PBOOL]
    _GetProcessDEPPolicy.restype = bool
    _GetProcessDEPPolicy.errcheck = RaiseIfZero
    lpFlags = DWORD(0)
    lpPermanent = BOOL(0)
    _GetProcessDEPPolicy(hProcess, byref(lpFlags), byref(lpPermanent))
    return (lpFlags.value, lpPermanent.value)