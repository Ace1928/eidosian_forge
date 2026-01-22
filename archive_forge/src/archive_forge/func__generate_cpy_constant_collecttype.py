import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_constant_collecttype(self, tp, name):
    is_int = tp.is_integer_type()
    if not is_int or self.target_is_python:
        self._do_collect_type(tp)