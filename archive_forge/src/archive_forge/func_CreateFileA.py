import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateFileA(lpFileName, dwDesiredAccess=GENERIC_ALL, dwShareMode=0, lpSecurityAttributes=None, dwCreationDisposition=OPEN_ALWAYS, dwFlagsAndAttributes=FILE_ATTRIBUTE_NORMAL, hTemplateFile=None):
    _CreateFileA = windll.kernel32.CreateFileA
    _CreateFileA.argtypes = [LPSTR, DWORD, DWORD, LPVOID, DWORD, DWORD, HANDLE]
    _CreateFileA.restype = HANDLE
    if not lpFileName:
        lpFileName = None
    if lpSecurityAttributes:
        lpSecurityAttributes = ctypes.pointer(lpSecurityAttributes)
    hFile = _CreateFileA(lpFileName, dwDesiredAccess, dwShareMode, lpSecurityAttributes, dwCreationDisposition, dwFlagsAndAttributes, hTemplateFile)
    if hFile == INVALID_HANDLE_VALUE:
        raise ctypes.WinError()
    return FileHandle(hFile)