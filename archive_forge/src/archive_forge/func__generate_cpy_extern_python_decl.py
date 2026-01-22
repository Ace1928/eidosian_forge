import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_extern_python_decl(self, tp, name):
    self._extern_python_decl(tp, name, 'static ')