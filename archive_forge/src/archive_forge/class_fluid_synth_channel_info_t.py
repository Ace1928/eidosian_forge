from ctypes import *
from ctypes.util import find_library
import os
class fluid_synth_channel_info_t(Structure):
    _fields_ = [('assigned', c_int), ('sfont_id', c_int), ('bank', c_int), ('program', c_int), ('name', c_char * 32), ('reserved', c_char * 32)]