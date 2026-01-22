import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_add_rec_def(a0, a1, a2, a3, a4, _elems=Elementaries(_lib.Z3_add_rec_def)):
    _elems.f(a0, a1, a2, a3, a4)
    _elems.Check(a0)