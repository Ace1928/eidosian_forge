import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_get_estimated_alloc_size(_elems=Elementaries(_lib.Z3_get_estimated_alloc_size)):
    r = _elems.f()
    return r