from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class ExtractElement(Instruction):

    def __init__(self, parent, vector, index, name=''):
        if not isinstance(vector.type, types.VectorType):
            raise TypeError('vector needs to be of VectorType.')
        if not isinstance(index.type, types.IntType):
            raise TypeError('index needs to be of IntType.')
        typ = vector.type.element
        super(ExtractElement, self).__init__(parent, typ, 'extractelement', [vector, index], name=name)

    def descr(self, buf):
        operands = ', '.join(('{0} {1}'.format(op.type, op.get_reference()) for op in self.operands))
        buf.append('{opname} {operands}\n'.format(opname=self.opname, operands=operands))