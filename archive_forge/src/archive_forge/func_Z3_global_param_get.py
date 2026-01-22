import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_global_param_get(a0, a1, _elems=Elementaries(_lib.Z3_global_param_get)):
    r = _elems.f(_str_to_bytes(a0), _str_to_bytes(a1))
    return r