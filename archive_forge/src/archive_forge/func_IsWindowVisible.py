from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def IsWindowVisible(hWnd):
    _IsWindowVisible = windll.user32.IsWindowVisible
    _IsWindowVisible.argtypes = [HWND]
    _IsWindowVisible.restype = bool
    return _IsWindowVisible(hWnd)