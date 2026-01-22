import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_open_log(a0, _elems=Elementaries(_lib.Z3_open_log)):
    r = _elems.f(_str_to_bytes(a0))
    return r