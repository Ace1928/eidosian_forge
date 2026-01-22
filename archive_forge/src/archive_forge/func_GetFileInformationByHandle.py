import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetFileInformationByHandle(hFile):
    _GetFileInformationByHandle = windll.kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = [HANDLE, LPBY_HANDLE_FILE_INFORMATION]
    _GetFileInformationByHandle.restype = bool
    _GetFileInformationByHandle.errcheck = RaiseIfZero
    lpFileInformation = BY_HANDLE_FILE_INFORMATION()
    _GetFileInformationByHandle(hFile, byref(lpFileInformation))
    return lpFileInformation