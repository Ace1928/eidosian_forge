import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
class MXCallbackList(ctypes.Structure):
    """Structure that holds Callback information. Passed to CustomOpProp."""
    _fields_ = [('num_callbacks', ctypes.c_int), ('callbacks', ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_int))), ('contexts', ctypes.POINTER(ctypes.c_void_p))]