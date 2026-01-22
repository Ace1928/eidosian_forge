import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetFinalPathNameByHandleA(hFile, dwFlags=FILE_NAME_NORMALIZED | VOLUME_NAME_DOS):
    _GetFinalPathNameByHandleA = windll.kernel32.GetFinalPathNameByHandleA
    _GetFinalPathNameByHandleA.argtypes = [HANDLE, LPSTR, DWORD, DWORD]
    _GetFinalPathNameByHandleA.restype = DWORD
    cchFilePath = _GetFinalPathNameByHandleA(hFile, None, 0, dwFlags)
    if cchFilePath == 0:
        raise ctypes.WinError()
    lpszFilePath = ctypes.create_string_buffer('', cchFilePath + 1)
    nCopied = _GetFinalPathNameByHandleA(hFile, lpszFilePath, cchFilePath, dwFlags)
    if nCopied <= 0 or nCopied > cchFilePath:
        raise ctypes.WinError()
    return lpszFilePath.value