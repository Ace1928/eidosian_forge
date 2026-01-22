import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _emit_bytecode_UnknownFloatType(self, tp, index):
    s = '_cffi_prim_float(sizeof(%s) *\n           (((%s)1) / 2) * 2 /* integer => 0, float => 1 */\n         )' % (tp.name, tp.name)
    self.cffi_types[index] = CffiOp(OP_PRIMITIVE, s)