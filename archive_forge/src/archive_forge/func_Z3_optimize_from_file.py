import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_optimize_from_file(a0, a1, a2, _elems=Elementaries(_lib.Z3_optimize_from_file)):
    _elems.f(a0, a1, _str_to_bytes(a2))
    _elems.Check(a0)