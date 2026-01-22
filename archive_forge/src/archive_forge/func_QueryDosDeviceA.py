import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def QueryDosDeviceA(lpDeviceName=None):
    _QueryDosDeviceA = windll.kernel32.QueryDosDeviceA
    _QueryDosDeviceA.argtypes = [LPSTR, LPSTR, DWORD]
    _QueryDosDeviceA.restype = DWORD
    _QueryDosDeviceA.errcheck = RaiseIfZero
    if not lpDeviceName:
        lpDeviceName = None
    ucchMax = 4096
    lpTargetPath = ctypes.create_string_buffer('', ucchMax)
    _QueryDosDeviceA(lpDeviceName, lpTargetPath, ucchMax)
    return lpTargetPath.value