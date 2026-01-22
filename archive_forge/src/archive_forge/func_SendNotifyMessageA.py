from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendNotifyMessageA(hWnd, Msg, wParam=0, lParam=0):
    _SendNotifyMessageA = windll.user32.SendNotifyMessageA
    _SendNotifyMessageA.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendNotifyMessageA.restype = bool
    _SendNotifyMessageA.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _SendNotifyMessageA(hWnd, Msg, wParam, lParam)