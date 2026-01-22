import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_rcf_num_to_decimal_string(a0, a1, a2, _elems=Elementaries(_lib.Z3_rcf_num_to_decimal_string)):
    r = _elems.f(a0, a1, a2)
    _elems.Check(a0)
    return _to_pystr(r)