from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_metricInfo_t(Structure):
    _fields_ = [('shortName', c_char_p), ('longName', c_char_p), ('unit', c_char_p)]