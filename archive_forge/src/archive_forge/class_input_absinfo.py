import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
class input_absinfo(Structure):
    _fields_ = [('value', c_int), ('minimum', c_int), ('maximum', c_int), ('fuzz', c_int), ('flat', c_int), ('resolution', c_int)]