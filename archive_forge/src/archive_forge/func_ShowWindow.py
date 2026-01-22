from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def ShowWindow(hWnd, nCmdShow=SW_SHOW):
    _ShowWindow = windll.user32.ShowWindow
    _ShowWindow.argtypes = [HWND, ctypes.c_int]
    _ShowWindow.restype = bool
    return _ShowWindow(hWnd, nCmdShow)