import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_FunctionPtrType(self, tp, index):
    raw = tp.as_raw_function()
    self.cffi_types[index] = CffiOp(OP_POINTER, self._typesdict[raw])