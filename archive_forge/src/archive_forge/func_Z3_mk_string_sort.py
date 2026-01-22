import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_mk_string_sort(a0, _elems=Elementaries(_lib.Z3_mk_string_sort)):
    r = _elems.f(a0)
    _elems.Check(a0)
    return r