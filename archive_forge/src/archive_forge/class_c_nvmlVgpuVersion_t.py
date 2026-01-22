from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuVersion_t(Structure):
    _fields_ = [('minVersion', c_uint), ('maxVersion', c_uint)]