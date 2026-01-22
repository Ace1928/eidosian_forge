from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class LandingPadInstr(Instruction):

    def __init__(self, parent, typ, name='', cleanup=False):
        super(LandingPadInstr, self).__init__(parent, typ, 'landingpad', [], name=name)
        self.cleanup = cleanup
        self.clauses = []

    def add_clause(self, clause):
        assert isinstance(clause, _LandingPadClause)
        self.clauses.append(clause)

    def descr(self, buf):
        fmt = 'landingpad {type}{cleanup}{clauses}\n'
        buf.append(fmt.format(type=self.type, cleanup=' cleanup' if self.cleanup else '', clauses=''.join(['\n      {0}'.format(clause) for clause in self.clauses])))