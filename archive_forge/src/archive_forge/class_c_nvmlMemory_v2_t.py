from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlMemory_v2_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('total', c_ulonglong), ('reserved', c_ulonglong), ('free', c_ulonglong), ('used', c_ulonglong)]
    _fmt_ = {'<default>': '%d B'}