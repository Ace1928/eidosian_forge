from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlVgpuSchedulerLog_t(_PrintableStructure):
    _fields_ = [('engineId', c_uint), ('schedulerPolicy', c_uint), ('arrMode', c_uint), ('schedulerParams', c_nvmlVgpuSchedulerParams_t), ('entriesCount', c_uint), ('logEntries', c_nvmlVgpuSchedulerLogEntry_t * NVML_SCHEDULER_SW_MAX_LOG_ENTRIES)]