from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlUnitFanSpeeds_t(_PrintableStructure):
    _fields_ = [('fans', c_nvmlUnitFanInfo_t * 24), ('count', c_uint)]