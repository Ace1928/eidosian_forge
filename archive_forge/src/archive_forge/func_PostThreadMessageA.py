from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def PostThreadMessageA(idThread, Msg, wParam=0, lParam=0):
    _PostThreadMessageA = windll.user32.PostThreadMessageA
    _PostThreadMessageA.argtypes = [DWORD, UINT, WPARAM, LPARAM]
    _PostThreadMessageA.restype = bool
    _PostThreadMessageA.errcheck = RaiseIfZero
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    _PostThreadMessageA(idThread, Msg, wParam, lParam)