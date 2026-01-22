import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
class timeval(Structure):
    _fields_ = [('tv_sec', c_ulong), ('tv_usec', c_ulong)]