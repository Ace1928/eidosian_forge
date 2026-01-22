from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class LoadAtomicInstr(Instruction):

    def __init__(self, parent, ptr, ordering, align, name=''):
        super(LoadAtomicInstr, self).__init__(parent, ptr.type.pointee, 'load atomic', [ptr], name=name)
        self.ordering = ordering
        self.align = align

    def descr(self, buf):
        [val] = self.operands
        buf.append('load atomic {0}, {1} {2} {3}, align {4}{5}\n'.format(val.type.pointee, val.type, val.get_reference(), self.ordering, self.align, self._stringify_metadata(leading_comma=True)))