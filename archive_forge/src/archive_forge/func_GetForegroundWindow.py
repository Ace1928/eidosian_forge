from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetForegroundWindow():
    _GetForegroundWindow = windll.user32.GetForegroundWindow
    _GetForegroundWindow.argtypes = []
    _GetForegroundWindow.restype = HWND
    _GetForegroundWindow.errcheck = RaiseIfZero
    return _GetForegroundWindow()