from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def FindWindowA(lpClassName=None, lpWindowName=None):
    _FindWindowA = windll.user32.FindWindowA
    _FindWindowA.argtypes = [LPSTR, LPSTR]
    _FindWindowA.restype = HWND
    hWnd = _FindWindowA(lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd