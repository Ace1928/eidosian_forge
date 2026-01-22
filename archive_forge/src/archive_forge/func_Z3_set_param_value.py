import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_set_param_value(a0, a1, a2, _elems=Elementaries(_lib.Z3_set_param_value)):
    _elems.f(a0, _str_to_bytes(a1), _str_to_bytes(a2))