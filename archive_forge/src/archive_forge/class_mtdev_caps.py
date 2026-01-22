import os
import time
from ctypes import cdll, Structure, c_ulong, c_int, c_ushort, \
class mtdev_caps(Structure):
    _fields_ = [('has_mtdata', c_int), ('has_slot', c_int), ('has_abs', c_int * MTDEV_ABS_SIZE), ('slot', input_absinfo), ('abs', input_absinfo * MTDEV_ABS_SIZE)]