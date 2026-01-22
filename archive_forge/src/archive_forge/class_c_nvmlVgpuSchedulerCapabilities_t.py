from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedulerCapabilities_t(_PrintableStructure):
    _fields_ = [('supportedSchedulers', c_uint * NVML_SUPPORTED_VGPU_SCHEDULER_POLICY_COUNT), ('maxTimeslice', c_uint), ('minTimeslice', c_uint), ('isArrModeSupported', c_uint), ('maxFrequencyForARR', c_uint), ('minFrequencyForARR', c_uint), ('maxAvgFactorForARR', c_uint), ('minAvgFactorForARR', c_uint)]