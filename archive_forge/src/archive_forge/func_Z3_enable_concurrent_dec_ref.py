import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_enable_concurrent_dec_ref(a0, _elems=Elementaries(_lib.Z3_enable_concurrent_dec_ref)):
    _elems.f(a0)
    _elems.Check(a0)