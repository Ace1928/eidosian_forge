from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlNvLinkPowerThres_t(Structure):
    _fields_ = [('lowPwrThreshold', c_uint)]