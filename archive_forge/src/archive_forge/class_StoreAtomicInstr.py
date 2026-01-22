from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class StoreAtomicInstr(Instruction):

    def __init__(self, parent, val, ptr, ordering, align):
        super(StoreAtomicInstr, self).__init__(parent, types.VoidType(), 'store atomic', [val, ptr])
        self.ordering = ordering
        self.align = align

    def descr(self, buf):
        val, ptr = self.operands
        buf.append('store atomic {0} {1}, {2} {3} {4}, align {5}{6}\n'.format(val.type, val.get_reference(), ptr.type, ptr.get_reference(), self.ordering, self.align, self._stringify_metadata(leading_comma=True)))