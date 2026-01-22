from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def GetWindowPlacement(hWnd):
    _GetWindowPlacement = windll.user32.GetWindowPlacement
    _GetWindowPlacement.argtypes = [HWND, PWINDOWPLACEMENT]
    _GetWindowPlacement.restype = bool
    _GetWindowPlacement.errcheck = RaiseIfZero
    lpwndpl = WINDOWPLACEMENT()
    lpwndpl.length = sizeof(lpwndpl)
    _GetWindowPlacement(hWnd, byref(lpwndpl))
    return WindowPlacement(lpwndpl)