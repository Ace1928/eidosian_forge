import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_parse_smtlib2_string(a0, a1, a2, a3, a4, a5, a6, a7, _elems=Elementaries(_lib.Z3_parse_smtlib2_string)):
    r = _elems.f(a0, _str_to_bytes(a1), a2, a3, a4, a5, a6, a7)
    _elems.Check(a0)
    return r