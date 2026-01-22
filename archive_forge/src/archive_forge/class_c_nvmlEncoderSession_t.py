from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlEncoderSession_t(_PrintableStructure):
    _fields_ = [('sessionId', c_uint), ('pid', c_uint), ('vgpuInstance', _nvmlVgpuInstance_t), ('codecType', c_uint), ('hResolution', c_uint), ('vResolution', c_uint), ('averageFps', c_uint), ('encodeLatency', c_uint)]