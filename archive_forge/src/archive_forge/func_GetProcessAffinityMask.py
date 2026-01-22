import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessAffinityMask(hProcess):
    _GetProcessAffinityMask = windll.kernel32.GetProcessAffinityMask
    _GetProcessAffinityMask.argtypes = [HANDLE, PDWORD_PTR, PDWORD_PTR]
    _GetProcessAffinityMask.restype = bool
    _GetProcessAffinityMask.errcheck = RaiseIfZero
    lpProcessAffinityMask = DWORD_PTR(0)
    lpSystemAffinityMask = DWORD_PTR(0)
    _GetProcessAffinityMask(hProcess, byref(lpProcessAffinityMask), byref(lpSystemAffinityMask))
    return (lpProcessAffinityMask.value, lpSystemAffinityMask.value)