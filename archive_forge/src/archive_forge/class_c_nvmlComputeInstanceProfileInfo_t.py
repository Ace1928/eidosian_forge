from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlComputeInstanceProfileInfo_t(Structure):
    _fields_ = [('id', c_uint), ('sliceCount', c_uint), ('instanceCount', c_uint), ('multiprocessorCount', c_uint), ('sharedCopyEngineCount', c_uint), ('sharedDecoderCount', c_uint), ('sharedEncoderCount', c_uint), ('sharedJpegCount', c_uint), ('sharedOfaCount', c_uint)]