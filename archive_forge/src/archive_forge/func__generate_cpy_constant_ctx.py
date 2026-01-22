import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_constant_ctx(self, tp, name):
    if not self.target_is_python and tp.is_integer_type():
        type_op = CffiOp(OP_CONSTANT_INT, -1)
    else:
        if self.target_is_python:
            const_kind = OP_DLOPEN_CONST
        else:
            const_kind = OP_CONSTANT
        type_index = self._typesdict[tp]
        type_op = CffiOp(const_kind, type_index)
    self._lsts['global'].append(GlobalExpr(name, '_cffi_const_%s' % name, type_op))