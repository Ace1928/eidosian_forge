import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_func_interp_get_else(a0, a1, _elems=Elementaries(_lib.Z3_func_interp_get_else)):
    r = _elems.f(a0, a1)
    _elems.Check(a0)
    return r