import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _enum_ctx(self, tp, cname):
    type_index = self._typesdict[tp]
    type_op = CffiOp(OP_ENUM, -1)
    if self.target_is_python:
        tp.check_not_partial()
    for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
        self._lsts['global'].append(GlobalExpr(enumerator, '_cffi_const_%s' % enumerator, type_op, check_value=enumvalue))
    if cname is not None and '$' not in cname and (not self.target_is_python):
        size = 'sizeof(%s)' % cname
        signed = '((%s)-1) <= 0' % cname
    else:
        basetp = tp.build_baseinttype(self.ffi, [])
        size = self.ffi.sizeof(basetp)
        signed = int(int(self.ffi.cast(basetp, -1)) < 0)
    allenums = ','.join(tp.enumerators)
    self._lsts['enum'].append(EnumExpr(tp.name, type_index, size, signed, allenums))