from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def UnDecorateSymbolNameW(DecoratedName, Flags=UNDNAME_COMPLETE):
    _UnDecorateSymbolNameW = windll.dbghelp.UnDecorateSymbolNameW
    _UnDecorateSymbolNameW.argtypes = [LPWSTR, LPWSTR, DWORD, DWORD]
    _UnDecorateSymbolNameW.restype = DWORD
    _UnDecorateSymbolNameW.errcheck = RaiseIfZero
    UndecoratedLength = _UnDecorateSymbolNameW(DecoratedName, None, 0, Flags)
    UnDecoratedName = ctypes.create_unicode_buffer(u'', UndecoratedLength + 1)
    _UnDecorateSymbolNameW(DecoratedName, UnDecoratedName, UndecoratedLength, Flags)
    return UnDecoratedName.value