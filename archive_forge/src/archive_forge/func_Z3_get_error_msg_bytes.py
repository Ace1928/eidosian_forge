import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_get_error_msg_bytes(a0, a1, _elems=Elementaries(_lib.Z3_get_error_msg)):
    r = _elems.f(a0, a1)
    _elems.Check(a0)
    return r