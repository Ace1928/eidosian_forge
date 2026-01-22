from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class LoadInstr(Instruction):

    def __init__(self, parent, ptr, name=''):
        super(LoadInstr, self).__init__(parent, ptr.type.pointee, 'load', [ptr], name=name)
        self.align = None

    def descr(self, buf):
        [val] = self.operands
        if self.align is not None:
            align = ', align %d' % self.align
        else:
            align = ''
        buf.append('load {0}, {1} {2}{3}{4}\n'.format(val.type.pointee, val.type, val.get_reference(), align, self._stringify_metadata(leading_comma=True)))