import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetConsoleCP():
    _GetConsoleCP = windll.kernel32.GetConsoleCP
    _GetConsoleCP.argytpes = []
    _GetConsoleCP.restype = UINT
    return _GetConsoleCP()