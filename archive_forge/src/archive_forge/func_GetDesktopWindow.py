from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetDesktopWindow():
    _GetDesktopWindow = windll.user32.GetDesktopWindow
    _GetDesktopWindow.argtypes = []
    _GetDesktopWindow.restype = HWND
    _GetDesktopWindow.errcheck = RaiseIfZero
    return _GetDesktopWindow()