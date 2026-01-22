import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _typedef_type(self, tp, name):
    return self._global_type(tp, '(*(%s *)0)' % (name,))