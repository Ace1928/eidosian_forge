from winappdbg.win32.defines import *
from winappdbg.win32.advapi32 import *
class WTS_CLIENT_DISPLAY(Structure):
    _fields_ = [('HorizontalResolution', DWORD), ('VerticalResolution', DWORD), ('ColorDepth', DWORD)]