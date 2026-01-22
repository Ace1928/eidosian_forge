import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_tactic_fail(a0, _elems=Elementaries(_lib.Z3_tactic_fail)):
    r = _elems.f(a0)
    _elems.Check(a0)
    return r