import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def Z3_optimize_register_model_eh(ctx, o, m, user_ctx, on_model_eh, _elems=Elementaries(_lib.Z3_optimize_register_model_eh)):
    _elems.f(ctx, o, m, user_ctx, on_model_eh)
    _elems.Check(ctx)