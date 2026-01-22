import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_del_config(a0, _elems=Elementaries(_lib.Z3_del_config)):
    _elems.f(a0)