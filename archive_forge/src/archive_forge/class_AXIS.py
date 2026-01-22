import ctypes
from ctypes.wintypes import HANDLE, BYTE, HWND, BOOL, UINT, LONG, WORD, DWORD, WCHAR, LPVOID
class AXIS(ctypes.Structure):
    _fields_ = (('axMin', LONG), ('axMax', LONG), ('axUnits', UINT), ('axResolution', FIX32))

    def get_scale(self):
        return 1 / float(self.axMax - self.axMin)

    def get_bias(self):
        return -self.axMin