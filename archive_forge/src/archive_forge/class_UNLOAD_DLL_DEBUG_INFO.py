import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class UNLOAD_DLL_DEBUG_INFO(Structure):
    _fields_ = [('lpBaseOfDll', LPVOID)]