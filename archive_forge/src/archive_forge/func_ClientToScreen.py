from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def ClientToScreen(hWnd, lpPoint):
    _ClientToScreen = windll.user32.ClientToScreen
    _ClientToScreen.argtypes = [HWND, LPPOINT]
    _ClientToScreen.restype = bool
    _ClientToScreen.errcheck = RaiseIfZero
    if isinstance(lpPoint, tuple):
        lpPoint = POINT(*lpPoint)
    else:
        lpPoint = POINT(lpPoint.x, lpPoint.y)
    _ClientToScreen(hWnd, byref(lpPoint))
    return Point(lpPoint.x, lpPoint.y)