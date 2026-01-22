from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
class __WindowEnumerator(object):
    """
    Window enumerator class. Used internally by the window enumeration APIs.
    """

    def __init__(self):
        self.hwnd = list()

    def __call__(self, hwnd, lParam):
        self.hwnd.append(hwnd)
        return TRUE