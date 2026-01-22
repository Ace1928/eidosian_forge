from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuProcessUtilizationSample_t(_PrintableStructure):
    _fields_ = [('vgpuInstance', _nvmlVgpuInstance_t), ('pid', c_uint), ('processName', c_char * NVML_VGPU_NAME_BUFFER_SIZE), ('timeStamp', c_ulonglong), ('smUtil', c_uint), ('memUtil', c_uint), ('encUtil', c_uint), ('decUtil', c_uint)]