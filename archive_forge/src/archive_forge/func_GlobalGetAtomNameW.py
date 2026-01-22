import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GlobalGetAtomNameW(nAtom):
    _GlobalGetAtomNameW = windll.kernel32.GlobalGetAtomNameW
    _GlobalGetAtomNameW.argtypes = [ATOM, LPWSTR, ctypes.c_int]
    _GlobalGetAtomNameW.restype = UINT
    _GlobalGetAtomNameW.errcheck = RaiseIfZero
    nSize = 64
    while 1:
        lpBuffer = ctypes.create_unicode_buffer(u'', nSize)
        nCopied = _GlobalGetAtomNameW(nAtom, lpBuffer, nSize)
        if nCopied < nSize - 1:
            break
        nSize = nSize + 64
    return lpBuffer.value