from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def IsWindow(hWnd):
    _IsWindow = windll.user32.IsWindow
    _IsWindow.argtypes = [HWND]
    _IsWindow.restype = bool
    return _IsWindow(hWnd)