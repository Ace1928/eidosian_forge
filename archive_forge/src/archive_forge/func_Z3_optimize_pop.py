import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_optimize_pop(a0, a1, _elems=Elementaries(_lib.Z3_optimize_pop)):
    _elems.f(a0, a1)
    _elems.Check(a0)