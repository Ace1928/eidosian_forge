from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlConfComputeSystemCaps_t(Structure):
    _fields_ = [('cpuCaps', c_uint), ('gpusCaps', c_uint)]