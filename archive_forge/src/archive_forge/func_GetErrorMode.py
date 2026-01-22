import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetErrorMode():
    _GetErrorMode = windll.kernel32.GetErrorMode
    _GetErrorMode.argtypes = []
    _GetErrorMode.restype = UINT
    return _GetErrorMode()