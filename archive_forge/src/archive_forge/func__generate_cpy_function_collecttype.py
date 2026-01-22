import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_function_collecttype(self, tp, name):
    self._do_collect_type(tp.as_raw_function())
    if tp.ellipsis and (not self.target_is_python):
        self._do_collect_type(tp)