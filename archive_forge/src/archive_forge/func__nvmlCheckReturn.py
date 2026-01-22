from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def _nvmlCheckReturn(ret):
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    return ret