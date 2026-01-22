from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class SwitchInstr(PredictableInstr, Terminator):

    def __init__(self, parent, opname, val, default):
        super(SwitchInstr, self).__init__(parent, opname, [val])
        self.default = default
        self.cases = []

    @property
    def value(self):
        return self.operands[0]

    def add_case(self, val, block):
        assert isinstance(block, Block)
        if not isinstance(val, Value):
            val = Constant(self.value.type, val)
        self.cases.append((val, block))

    def descr(self, buf):
        cases = ['{0} {1}, label {2}'.format(val.type, val.get_reference(), blk.get_reference()) for val, blk in self.cases]
        buf.append('switch {0} {1}, label {2} [{3}]  {4}\n'.format(self.value.type, self.value.get_reference(), self.default.get_reference(), ' '.join(cases), self._stringify_metadata(leading_comma=True)))