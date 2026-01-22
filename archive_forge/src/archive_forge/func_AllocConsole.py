import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def AllocConsole():
    _AllocConsole = windll.kernel32.AllocConsole
    _AllocConsole.argytpes = []
    _AllocConsole.restype = bool
    _AllocConsole.errcheck = RaiseIfZero
    _AllocConsole()