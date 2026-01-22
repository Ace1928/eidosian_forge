from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendMessageA(hWnd, Msg, wParam=0, lParam=0):
    _SendMessageA = windll.user32.SendMessageA
    _SendMessageA.argtypes = [HWND, UINT, WPARAM, LPARAM]
    _SendMessageA.restype = LRESULT
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendMessageA(hWnd, Msg, wParam, lParam)