import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetSystemTimeAsFileTime():
    _GetSystemTimeAsFileTime = windll.kernel32.GetSystemTimeAsFileTime
    _GetSystemTimeAsFileTime.argtypes = [LPFILETIME]
    _GetSystemTimeAsFileTime.restype = None
    FileTime = FILETIME()
    _GetSystemTimeAsFileTime(byref(FileTime))
    return FileTime