from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SetForegroundWindow(hWnd):
    _SetForegroundWindow = windll.user32.SetForegroundWindow
    _SetForegroundWindow.argtypes = [HWND]
    _SetForegroundWindow.restype = bool
    _SetForegroundWindow.errcheck = RaiseIfZero
    return _SetForegroundWindow(hWnd)