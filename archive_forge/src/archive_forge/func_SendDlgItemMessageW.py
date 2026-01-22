from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def SendDlgItemMessageW(hDlg, nIDDlgItem, Msg, wParam=0, lParam=0):
    _SendDlgItemMessageW = windll.user32.SendDlgItemMessageW
    _SendDlgItemMessageW.argtypes = [HWND, ctypes.c_int, UINT, WPARAM, LPARAM]
    _SendDlgItemMessageW.restype = LRESULT
    wParam = MAKE_WPARAM(wParam)
    lParam = MAKE_LPARAM(lParam)
    return _SendDlgItemMessageW(hDlg, nIDDlgItem, Msg, wParam, lParam)