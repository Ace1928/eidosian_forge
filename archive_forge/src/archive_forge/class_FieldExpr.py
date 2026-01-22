import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
class FieldExpr:

    def __init__(self, name, field_offset, field_size, fbitsize, field_type_op):
        self.name = name
        self.field_offset = field_offset
        self.field_size = field_size
        self.fbitsize = fbitsize
        self.field_type_op = field_type_op

    def as_c_expr(self):
        spaces = ' ' * len(self.name)
        return '  { "%s", %s,\n' % (self.name, self.field_offset) + '     %s   %s,\n' % (spaces, self.field_size) + '     %s   %s },' % (spaces, self.field_type_op.as_c_expr())

    def as_python_expr(self):
        raise NotImplementedError

    def as_field_python_expr(self):
        if self.field_type_op.op == OP_NOOP:
            size_expr = ''
        elif self.field_type_op.op == OP_BITFIELD:
            size_expr = format_four_bytes(self.fbitsize)
        else:
            raise NotImplementedError
        return "b'%s%s%s'" % (self.field_type_op.as_python_bytes(), size_expr, self.name)