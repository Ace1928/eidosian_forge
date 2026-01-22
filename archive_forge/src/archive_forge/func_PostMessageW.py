from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def PostMessageW(hWnd, Msg, wParam=0, lParam=0):
    _PostMessageW = windll.user32.PostMessageW
    _PostMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _PostMessageW.restype = bool
    _PostMessageW.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostMessageW(hWnd, Msg, wParam, lParam)