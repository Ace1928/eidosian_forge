from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuInstanceUtilizationSample_t(_PrintableStructure):
    _fields_ = [('vgpuInstance', _nvmlVgpuInstance_t), ('timeStamp', c_ulonglong), ('smUtil', c_nvmlValue_t), ('memUtil', c_nvmlValue_t), ('encUtil', c_nvmlValue_t), ('decUtil', c_nvmlValue_t)]