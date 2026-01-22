from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlSample_t(_PrintableStructure):
    _fields_ = [('timeStamp', c_ulonglong), ('sampleValue', c_nvmlValue_t)]