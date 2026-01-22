from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendMessageTimeoutW(hWnd, Msg, wParam=0, lParam=0):
    _SendMessageTimeoutW = windll.user32.SendMessageTimeoutW
    _SendMessageTimeoutW.argtypes = [HWND, UINT, WPARAM, LPARAM, UINT, UINT, PDWORD_PTR]
    _SendMessageTimeoutW.restype = LRESULT
    _SendMessageTimeoutW.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    dwResult = DWORD(0)
    _SendMessageTimeoutW(hWnd, Msg, wParam, lParam, fuFlags, uTimeout, byref(dwResult))
    return dwResult.value