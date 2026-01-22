from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class CmpXchg(Instruction):
    """This instruction has changed since llvm3.5.  It is not compatible with
    older llvm versions.
    """

    def __init__(self, parent, ptr, cmp, val, ordering, failordering, name):
        outtype = types.LiteralStructType([val.type, types.IntType(1)])
        super(CmpXchg, self).__init__(parent, outtype, 'cmpxchg', (ptr, cmp, val), name=name)
        self.ordering = ordering
        self.failordering = failordering

    def descr(self, buf):
        ptr, cmpval, val = self.operands
        fmt = 'cmpxchg {ptrty} {ptr}, {ty} {cmp}, {ty} {val} {ordering} {failordering} {metadata}\n'
        buf.append(fmt.format(ptrty=ptr.type, ptr=ptr.get_reference(), ty=cmpval.type, cmp=cmpval.get_reference(), val=val.get_reference(), ordering=self.ordering, failordering=self.failordering, metadata=self._stringify_metadata(leading_comma=True)))