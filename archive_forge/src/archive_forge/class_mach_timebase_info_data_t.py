import ctypes
import ctypes.util
import os
import sys
import threading
import time
class mach_timebase_info_data_t(ctypes.Structure):
    """System timebase info. Defined in <mach/mach_time.h>."""
    _fields_ = (('numer', ctypes.c_uint32), ('denom', ctypes.c_uint32))