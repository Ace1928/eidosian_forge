from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def SymSetOptions(SymOptions):
    _SymSetOptions = windll.dbghelp.SymSetOptions
    _SymSetOptions.argtypes = [DWORD]
    _SymSetOptions.restype = DWORD
    _SymSetOptions.errcheck = RaiseIfZero
    _SymSetOptions(SymOptions)