from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlUnitInfo_t(_PrintableStructure):
    _fields_ = [('name', c_char * 96), ('id', c_char * 96), ('serial', c_char * 96), ('firmwareVersion', c_char * 96)]