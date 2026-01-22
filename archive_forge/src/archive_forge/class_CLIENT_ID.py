from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class CLIENT_ID(Structure):
    _fields_ = [('UniqueProcess', PVOID), ('UniqueThread', PVOID)]