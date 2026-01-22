import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class DIValue(NamedValue):
    """
    A debug information descriptor, containing key-value pairs.

    Do not instantiate directly, use Module.add_debug_info() instead.
    """
    name_prefix = '!'

    def __init__(self, parent, is_distinct, kind, operands, name):
        super(DIValue, self).__init__(parent, types.MetaDataType(), name=name)
        self.is_distinct = is_distinct
        self.kind = kind
        self.operands = tuple(operands)
        parent.metadata.append(self)

    def descr(self, buf):
        if self.is_distinct:
            buf += ('distinct ',)
        operands = []
        for key, value in self.operands:
            if value is None:
                strvalue = 'null'
            elif value is True:
                strvalue = 'true'
            elif value is False:
                strvalue = 'false'
            elif isinstance(value, DIToken):
                strvalue = value.value
            elif isinstance(value, str):
                strvalue = '"{}"'.format(_escape_string(value))
            elif isinstance(value, int):
                strvalue = str(value)
            elif isinstance(value, NamedValue):
                strvalue = value.get_reference()
            else:
                raise TypeError('invalid operand type for debug info: %r' % (value,))
            operands.append('{0}: {1}'.format(key, strvalue))
        operands = ', '.join(operands)
        buf += ('!', self.kind, '(', operands, ')\n')

    def _get_reference(self):
        return self.name_prefix + str(self.name)

    def __eq__(self, other):
        if isinstance(other, DIValue):
            return self.is_distinct == other.is_distinct and self.kind == other.kind and (self.operands == other.operands)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.is_distinct, self.kind, self.operands))