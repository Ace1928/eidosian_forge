from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def ShowWindowAsync(hWnd, nCmdShow=SW_SHOW):
    _ShowWindowAsync = windll.user32.ShowWindowAsync
    _ShowWindowAsync.argtypes = [HWND, ctypes.c_int]
    _ShowWindowAsync.restype = bool
    return _ShowWindowAsync(hWnd, nCmdShow)