import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
class Elementaries:

    def __init__(self, f):
        self.f = f
        self.get_error_code = _lib.Z3_get_error_code
        self.get_error_message = _lib.Z3_get_error_msg
        self.OK = Z3_OK
        self.Exception = Z3Exception

    def Check(self, ctx):
        err = self.get_error_code(ctx)
        if err != self.OK:
            raise self.Exception(self.get_error_message(ctx, err))