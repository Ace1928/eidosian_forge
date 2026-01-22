from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class TEB_ACTIVE_FRAME_CONTEXT(Structure):
    _fields_ = [('Flags', DWORD), ('FrameName', LPVOID)]