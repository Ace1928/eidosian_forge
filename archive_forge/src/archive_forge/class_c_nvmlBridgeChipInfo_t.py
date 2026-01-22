from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlBridgeChipInfo_t(_PrintableStructure):
    _fields_ = [('type', _nvmlBridgeChipType_t), ('fwVersion', c_uint)]