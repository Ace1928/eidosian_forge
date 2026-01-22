from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendDlgItemMessageA(hDlg, nIDDlgItem, Msg, wParam=0, lParam=0):
    _SendDlgItemMessageA = windll.user32.SendDlgItemMessageA
    _SendDlgItemMessageA.argtypes = [HWND, ctypes.c_int, UINT, WPARAM, LPARAM]
    _SendDlgItemMessageA.restype = LRESULT
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendDlgItemMessageA(hDlg, nIDDlgItem, Msg, wParam, lParam)