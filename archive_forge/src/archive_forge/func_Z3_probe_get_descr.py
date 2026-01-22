import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_probe_get_descr(a0, a1, _elems=Elementaries(_lib.Z3_probe_get_descr)):
    r = _elems.f(a0, _str_to_bytes(a1))
    _elems.Check(a0)
    return _to_pystr(r)