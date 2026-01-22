from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetWindowLongW(hWnd, nIndex, dwNewLong):
    _SetWindowLongW = windll.user32.SetWindowLongW
    _SetWindowLongW.argtypes = [HWND, ctypes.c_int, DWORD]
    _SetWindowLongW.restype = DWORD
    SetLastError(ERROR_SUCCESS)
    retval = _SetWindowLongW(hWnd, nIndex, dwNewLong)
    if retval == 0:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return retval