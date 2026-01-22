import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class MDValue(NamedValue):
    """
    A metadata node's value, consisting of a sequence of elements ("operands").

    Do not instantiate directly, use Module.add_metadata() instead.
    """
    name_prefix = '!'

    def __init__(self, parent, values, name):
        super(MDValue, self).__init__(parent, types.MetaDataType(), name=name)
        self.operands = tuple(values)
        parent.metadata.append(self)

    def descr(self, buf):
        operands = []
        for op in self.operands:
            if isinstance(op.type, types.MetaDataType):
                if isinstance(op, Constant) and op.constant is None:
                    operands.append('null')
                else:
                    operands.append(op.get_reference())
            else:
                operands.append('{0} {1}'.format(op.type, op.get_reference()))
        operands = ', '.join(operands)
        buf += ('!{{ {0} }}'.format(operands), '\n')

    def _get_reference(self):
        return self.name_prefix + str(self.name)

    def __eq__(self, other):
        if isinstance(other, MDValue):
            return self.operands == other.operands
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.operands)