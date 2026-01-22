import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def AttachConsole(dwProcessId=ATTACH_PARENT_PROCESS):
    _AttachConsole = windll.kernel32.AttachConsole
    _AttachConsole.argytpes = [DWORD]
    _AttachConsole.restype = bool
    _AttachConsole.errcheck = RaiseIfZero
    _AttachConsole(dwProcessId)