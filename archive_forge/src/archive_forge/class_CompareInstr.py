from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class CompareInstr(Instruction):
    OPNAME = 'invalid-compare'
    VALID_OP = {}

    def __init__(self, parent, op, lhs, rhs, name='', flags=[]):
        if op not in self.VALID_OP:
            raise ValueError('invalid comparison %r for %s' % (op, self.OPNAME))
        for flag in flags:
            if flag not in self.VALID_FLAG:
                raise ValueError('invalid flag %r for %s' % (flag, self.OPNAME))
        opname = self.OPNAME
        if isinstance(lhs.type, types.VectorType):
            typ = types.VectorType(types.IntType(1), lhs.type.count)
        else:
            typ = types.IntType(1)
        super(CompareInstr, self).__init__(parent, typ, opname, [lhs, rhs], flags=flags, name=name)
        self.op = op

    def descr(self, buf):
        buf.append('{opname}{flags} {op} {ty} {lhs}, {rhs} {meta}\n'.format(opname=self.opname, flags=''.join((' ' + it for it in self.flags)), op=self.op, ty=self.operands[0].type, lhs=self.operands[0].get_reference(), rhs=self.operands[1].get_reference(), meta=self._stringify_metadata(leading_comma=True)))