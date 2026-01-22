import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_anonymous_collecttype(self, tp, name):
    if isinstance(tp, model.EnumType):
        self._generate_cpy_enum_collecttype(tp, name)
    else:
        self._struct_collecttype(tp)