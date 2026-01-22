import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetFullPathNameA(lpFileName):
    _GetFullPathNameA = windll.kernel32.GetFullPathNameA
    _GetFullPathNameA.argtypes = [LPSTR, DWORD, LPSTR, POINTER(LPSTR)]
    _GetFullPathNameA.restype = DWORD
    nBufferLength = _GetFullPathNameA(lpFileName, 0, None, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_string_buffer('', nBufferLength + 1)
    lpFilePart = LPSTR()
    nCopied = _GetFullPathNameA(lpFileName, nBufferLength, lpBuffer, byref(lpFilePart))
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return (lpBuffer.value, lpFilePart.value)