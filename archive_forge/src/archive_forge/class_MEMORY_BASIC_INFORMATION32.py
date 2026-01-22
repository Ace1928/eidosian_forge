import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class MEMORY_BASIC_INFORMATION32(Structure):
    _fields_ = [('BaseAddress', DWORD), ('AllocationBase', DWORD), ('AllocationProtect', DWORD), ('RegionSize', DWORD), ('State', DWORD), ('Protect', DWORD), ('Type', DWORD)]