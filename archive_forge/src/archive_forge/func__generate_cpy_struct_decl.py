import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_struct_decl(self, tp, name):
    self._struct_decl(tp, *self._struct_names(tp))