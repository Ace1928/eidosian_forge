import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_PointerType(self, tp, index):
    self.cffi_types[index] = CffiOp(OP_POINTER, self._typesdict[tp.totype])