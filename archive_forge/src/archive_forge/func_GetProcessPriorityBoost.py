import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessPriorityBoost(hProcess):
    _GetProcessPriorityBoost = windll.kernel32.GetProcessPriorityBoost
    _GetProcessPriorityBoost.argtypes = [HANDLE, PBOOL]
    _GetProcessPriorityBoost.restype = bool
    _GetProcessPriorityBoost.errcheck = RaiseIfZero
    pDisablePriorityBoost = BOOL(False)
    _GetProcessPriorityBoost(hProcess, byref(pDisablePriorityBoost))
    return bool(pDisablePriorityBoost.value)