from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def PostMessageA(hWnd, Msg, wParam=0, lParam=0):
    _PostMessageA = windll.user32.PostMessageA
    _PostMessageA.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _PostMessageA.restype = bool
    _PostMessageA.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostMessageA(hWnd, Msg, wParam, lParam)