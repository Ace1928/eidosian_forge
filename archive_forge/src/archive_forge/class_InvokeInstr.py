from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class InvokeInstr(CallInstr):

    def __init__(self, parent, func, args, normal_to, unwind_to, name='', cconv=None, fastmath=(), attrs=(), arg_attrs=None):
        assert isinstance(normal_to, Block)
        assert isinstance(unwind_to, Block)
        super(InvokeInstr, self).__init__(parent, func, args, name, cconv, tail=False, fastmath=fastmath, attrs=attrs, arg_attrs=arg_attrs)
        self.opname = 'invoke'
        self.normal_to = normal_to
        self.unwind_to = unwind_to

    def descr(self, buf):
        super(InvokeInstr, self)._descr(buf, add_metadata=False)
        buf.append('      to label {0} unwind label {1}{metadata}\n'.format(self.normal_to.get_reference(), self.unwind_to.get_reference(), metadata=self._stringify_metadata(leading_comma=True)))