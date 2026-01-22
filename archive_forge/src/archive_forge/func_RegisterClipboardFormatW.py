from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def RegisterClipboardFormatW(lpString):
    _RegisterClipboardFormatW = windll.user32.RegisterClipboardFormatW
    _RegisterClipboardFormatW.argtypes = [LPWSTR]
    _RegisterClipboardFormatW.restype = UINT
    _RegisterClipboardFormatW.errcheck = RaiseIfZero
    return _RegisterClipboardFormatW(lpString)