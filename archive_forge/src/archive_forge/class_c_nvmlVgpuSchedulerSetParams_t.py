from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedulerSetParams_t(Union):
    _fields_ = [('vgpuSchedDataWithARR', c_nvmlVgpuSchedSetDataWithARR_t), ('vgpuSchedData', c_nvmlVgpuSchedSetData_t)]