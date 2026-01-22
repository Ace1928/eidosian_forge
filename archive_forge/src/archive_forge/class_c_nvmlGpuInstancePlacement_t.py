from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuInstancePlacement_t(Structure):
    _fields_ = [('start', c_uint), ('size', c_uint)]