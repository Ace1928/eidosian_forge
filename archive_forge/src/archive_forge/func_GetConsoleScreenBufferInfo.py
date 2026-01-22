import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetConsoleScreenBufferInfo(hConsoleOutput=None):
    _GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
    _GetConsoleScreenBufferInfo.argytpes = [HANDLE, PCONSOLE_SCREEN_BUFFER_INFO]
    _GetConsoleScreenBufferInfo.restype = bool
    _GetConsoleScreenBufferInfo.errcheck = RaiseIfZero
    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    ConsoleScreenBufferInfo = CONSOLE_SCREEN_BUFFER_INFO()
    _GetConsoleScreenBufferInfo(hConsoleOutput, byref(ConsoleScreenBufferInfo))
    return ConsoleScreenBufferInfo