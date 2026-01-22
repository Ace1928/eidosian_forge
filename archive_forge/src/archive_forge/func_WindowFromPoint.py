from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def WindowFromPoint(point):
    _WindowFromPoint = windll.user32.WindowFromPoint
    _WindowFromPoint.argtypes = [POINT]
    _WindowFromPoint.restype = HWND
    _WindowFromPoint.errcheck = RaiseIfZero
    if isinstance(point, tuple):
        point = POINT(*point)
    return _WindowFromPoint(point)