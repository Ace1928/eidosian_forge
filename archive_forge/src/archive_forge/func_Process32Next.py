import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Process32Next(hSnapshot, pe=None):
    _Process32Next = windll.kernel32.Process32Next
    _Process32Next.argtypes = [HANDLE, LPPROCESSENTRY32]
    _Process32Next.restype = bool
    if pe is None:
        pe = PROCESSENTRY32()
    pe.dwSize = sizeof(PROCESSENTRY32)
    success = _Process32Next(hSnapshot, byref(pe))
    if not success:
        if GetLastError() == ERROR_NO_MORE_FILES:
            return None
        raise ctypes.WinError()
    return pe