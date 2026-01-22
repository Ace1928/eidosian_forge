from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def FindWindowExW(hwndParent=None, hwndChildAfter=None, lpClassName=None, lpWindowName=None):
    _FindWindowExW = windll.user32.FindWindowExW
    _FindWindowExW.argtypes = [HWND, HWND, LPWSTR, LPWSTR]
    _FindWindowExW.restype = HWND
    hWnd = _FindWindowExW(hwndParent, hwndChildAfter, lpClassName, lpWindowName)
    if not hWnd:
        errcode = GetLastError()
        if errcode != ERROR_SUCCESS:
            raise ctypes.WinError(errcode)
    return hWnd