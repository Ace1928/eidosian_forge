import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
class input_event(Structure):
    _fields_ = [('time', timeval), ('type', c_ushort), ('code', c_ushort), ('value', c_int)]