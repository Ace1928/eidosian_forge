from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class ShuffleVector(Instruction):

    def __init__(self, parent, vector1, vector2, mask, name=''):
        if not isinstance(vector1.type, types.VectorType):
            raise TypeError('vector1 needs to be of VectorType.')
        if vector2 != Undefined:
            if vector2.type != vector1.type:
                raise TypeError('vector2 needs to be ' + 'Undefined or of the same type as vector1.')
        if not isinstance(mask, Constant) or not isinstance(mask.type, types.VectorType) or (not (isinstance(mask.type.element, types.IntType) and mask.type.element.width == 32)):
            raise TypeError('mask needs to be a constant i32 vector.')
        typ = types.VectorType(vector1.type.element, mask.type.count)
        index_range = range(vector1.type.count if vector2 == Undefined else 2 * vector1.type.count)
        if not all((ii.constant in index_range for ii in mask.constant)):
            raise IndexError('mask values need to be in {0}'.format(index_range))
        super(ShuffleVector, self).__init__(parent, typ, 'shufflevector', [vector1, vector2, mask], name=name)

    def descr(self, buf):
        buf.append('shufflevector {0} {1}\n'.format(', '.join(('{0} {1}'.format(op.type, op.get_reference()) for op in self.operands)), self._stringify_metadata(leading_comma=True)))