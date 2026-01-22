from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetShellWindow():
    _GetShellWindow = windll.user32.GetShellWindow
    _GetShellWindow.argtypes = []
    _GetShellWindow.restype = HWND
    _GetShellWindow.errcheck = RaiseIfZero
    return _GetShellWindow()