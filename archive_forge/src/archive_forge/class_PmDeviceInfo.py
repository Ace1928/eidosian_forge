import ctypes
import ctypes.util
import sys
class PmDeviceInfo(ctypes.Structure):
    _fields_ = [('structVersion', ctypes.c_int), ('interface', ctypes.c_char_p), ('name', ctypes.c_char_p), ('is_input', ctypes.c_int), ('is_output', ctypes.c_int), ('opened', ctypes.c_int)]