import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_constant_decl(self, tp, name):
    is_int = tp.is_integer_type()
    self._generate_cpy_const(is_int, name, tp)