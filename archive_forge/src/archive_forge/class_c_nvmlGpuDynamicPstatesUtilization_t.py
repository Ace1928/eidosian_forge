from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpuDynamicPstatesUtilization_t(Structure):
    _fields_ = [('bIsPresent', c_uint, 1), ('percentage', c_uint), ('incThreshold', c_uint), ('decThreshold', c_uint)]