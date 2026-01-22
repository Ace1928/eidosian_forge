import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_rcf_mk_pi(a0, _elems=Elementaries(_lib.Z3_rcf_mk_pi)):
    r = _elems.f(a0)
    _elems.Check(a0)
    return r