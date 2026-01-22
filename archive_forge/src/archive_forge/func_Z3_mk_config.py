import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_mk_config(_elems=Elementaries(_lib.Z3_mk_config)):
    r = _elems.f()
    return r