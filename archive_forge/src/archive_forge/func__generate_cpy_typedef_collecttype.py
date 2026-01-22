import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_typedef_collecttype(self, tp, name):
    self._do_collect_type(self._typedef_type(tp, name))