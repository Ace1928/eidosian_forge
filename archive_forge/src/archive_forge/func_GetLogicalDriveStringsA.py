import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetLogicalDriveStringsA():
    _GetLogicalDriveStringsA = ctypes.windll.kernel32.GetLogicalDriveStringsA
    _GetLogicalDriveStringsA.argtypes = [DWORD, LPSTR]
    _GetLogicalDriveStringsA.restype = DWORD
    _GetLogicalDriveStringsA.errcheck = RaiseIfZero
    nBufferLength = 4 * 26 + 1
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    _GetLogicalDriveStringsA(nBufferLength, lpBuffer)
    drive_strings = list()
    string_p = addressof(lpBuffer)
    sizeof_char = sizeof(ctypes.c_char)
    while True:
        string_v = ctypes.string_at(string_p)
        if string_v == '':
            break
        drive_strings.append(string_v)
        string_p += len(string_v) + sizeof_char
    return drive_strings