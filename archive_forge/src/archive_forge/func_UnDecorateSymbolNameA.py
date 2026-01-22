from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def UnDecorateSymbolNameA(DecoratedName, Flags=UNDNAME_COMPLETE):
    _UnDecorateSymbolNameA = windll.dbghelp.UnDecorateSymbolName
    _UnDecorateSymbolNameA.argtypes = [LPSTR, LPSTR, DWORD, DWORD]
    _UnDecorateSymbolNameA.restype = DWORD
    _UnDecorateSymbolNameA.errcheck = RaiseIfZero
    UndecoratedLength = _UnDecorateSymbolNameA(DecoratedName, None, 0, Flags)
    UnDecoratedName = ctypes.create_string_buffer('', UndecoratedLength + 1)
    _UnDecorateSymbolNameA(DecoratedName, UnDecoratedName, UndecoratedLength, Flags)
    return UnDecoratedName.value