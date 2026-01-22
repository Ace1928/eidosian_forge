from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class PhiInstr(Instruction):

    def __init__(self, parent, typ, name, flags=()):
        super(PhiInstr, self).__init__(parent, typ, 'phi', (), name=name, flags=flags)
        self.incomings = []

    def descr(self, buf):
        incs = ', '.join(('[{0}, {1}]'.format(v.get_reference(), b.get_reference()) for v, b in self.incomings))
        buf.append('phi {0} {1} {2} {3}\n'.format(' '.join(self.flags), self.type, incs, self._stringify_metadata(leading_comma=True)))

    def add_incoming(self, value, block):
        assert isinstance(block, Block)
        self.incomings.append((value, block))

    def replace_usage(self, old, new):
        self.incomings = [(new if val is old else val, blk) for val, blk in self.incomings]