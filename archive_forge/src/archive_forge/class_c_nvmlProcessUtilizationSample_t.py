from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlProcessUtilizationSample_t(_PrintableStructure):
    _fields_ = [('pid', c_uint), ('timeStamp', c_ulonglong), ('smUtil', c_uint), ('memUtil', c_uint), ('encUtil', c_uint), ('decUtil', c_uint)]