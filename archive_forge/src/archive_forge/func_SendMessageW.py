from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendMessageW(hWnd, Msg, wParam=0, lParam=0):
    _SendMessageW = windll.user32.SendMessageW
    _SendMessageW.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendMessageW.restype = LRESULT
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendMessageW(hWnd, Msg, wParam, lParam)