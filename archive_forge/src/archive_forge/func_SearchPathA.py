import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SearchPathA(lpPath, lpFileName, lpExtension):
    _SearchPathA = windll.kernel32.SearchPathA
    _SearchPathA.argtypes = [LPSTR, LPSTR, LPSTR, DWORD, LPSTR, POINTER(LPSTR)]
    _SearchPathA.restype = DWORD
    _SearchPathA.errcheck = RaiseIfZero
    if not lpPath:
        lpPath = None
    if not lpExtension:
        lpExtension = None
    nBufferLength = _SearchPathA(lpPath, lpFileName, lpExtension, 0, None, None)
    lpBuffer = ctypes.create_string_buffer('', nBufferLength + 1)
    lpFilePart = LPSTR()
    _SearchPathA(lpPath, lpFileName, lpExtension, nBufferLength, lpBuffer, byref(lpFilePart))
    lpFilePart = lpFilePart.value
    lpBuffer = lpBuffer.value
    if lpBuffer == '':
        if GetLastError() == ERROR_SUCCESS:
            raise ctypes.WinError(ERROR_FILE_NOT_FOUND)
        raise ctypes.WinError()
    return (lpBuffer, lpFilePart)