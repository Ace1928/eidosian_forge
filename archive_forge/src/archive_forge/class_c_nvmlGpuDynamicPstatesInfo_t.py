from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuDynamicPstatesInfo_t(Structure):
    _fields_ = [('flags', c_uint), ('utilization', c_nvmlGpuDynamicPstatesUtilization_t * NVML_MAX_GPU_UTILIZATIONS)]