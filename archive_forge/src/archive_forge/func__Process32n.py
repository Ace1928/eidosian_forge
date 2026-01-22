import os
from ctypes import (
from ctypes.wintypes import DWORD, LONG
def _Process32n(fun, hSnapshot, pe=None):
    if pe is None:
        pe = PROCESSENTRY32()
    pe.dwSize = sizeof(PROCESSENTRY32)
    success = fun(hSnapshot, byref(pe))
    if not success:
        if windll.kernel32.GetLastError() == ERROR_NO_MORE_FILES:
            return
        raise WinError()
    return pe