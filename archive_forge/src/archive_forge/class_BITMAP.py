from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import GetLastError, SetLastError
class BITMAP(Structure):
    _fields_ = [('bmType', LONG), ('bmWidth', LONG), ('bmHeight', LONG), ('bmWidthBytes', LONG), ('bmPlanes', WORD), ('bmBitsPixel', WORD), ('bmBits', LPVOID)]