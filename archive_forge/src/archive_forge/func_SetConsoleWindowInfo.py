import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetConsoleWindowInfo(hConsoleOutput, bAbsolute, lpConsoleWindow):
    _SetConsoleWindowInfo = windll.kernel32.SetConsoleWindowInfo
    _SetConsoleWindowInfo.argytpes = [HANDLE, BOOL, PSMALL_RECT]
    _SetConsoleWindowInfo.restype = bool
    _SetConsoleWindowInfo.errcheck = RaiseIfZero
    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    if isinstance(lpConsoleWindow, SMALL_RECT):
        ConsoleWindow = lpConsoleWindow
    else:
        ConsoleWindow = SMALL_RECT(*lpConsoleWindow)
    _SetConsoleWindowInfo(hConsoleOutput, bAbsolute, byref(ConsoleWindow))