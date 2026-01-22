import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetFullPathNameW(lpFileName):
    _GetFullPathNameW = windll.kernel32.GetFullPathNameW
    _GetFullPathNameW.argtypes = [LPWSTR, DWORD, LPWSTR, POINTER(LPWSTR)]
    _GetFullPathNameW.restype = DWORD
    nBufferLength = _GetFullPathNameW(lpFileName, 0, None, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength + 1)
    lpFilePart = LPWSTR()
    nCopied = _GetFullPathNameW(lpFileName, nBufferLength, lpBuffer, byref(lpFilePart))
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return (lpBuffer.value, lpFilePart.value)