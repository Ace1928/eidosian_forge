import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def as_python_expr(self):
    return "b'%s%s'" % (format_four_bytes(self.type_index), self.name)