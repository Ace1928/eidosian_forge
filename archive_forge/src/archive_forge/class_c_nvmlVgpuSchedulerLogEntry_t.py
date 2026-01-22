from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedulerLogEntry_t(_PrintableStructure):
    _fields_ = [('timestamp', c_ulonglong), ('timeRunTotal', c_ulonglong), ('timeRun', c_ulonglong), ('swRunlistId', c_uint), ('targetTimeSlice', c_ulonglong), ('cumulativePreemptionTime', c_ulonglong)]