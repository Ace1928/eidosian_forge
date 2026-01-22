from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedDataWithARR_t(_PrintableStructure):
    _fields_ = [('avgFactor', c_uint), ('timeslice', c_uint)]