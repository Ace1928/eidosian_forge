import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class MetaDataString(NamedValue):
    """
    A metadata string, i.e. a constant string used as a value in a metadata
    node.
    """

    def __init__(self, parent, string):
        super(MetaDataString, self).__init__(parent, types.MetaDataType(), name='')
        self.string = string

    def descr(self, buf):
        buf += (self.get_reference(), '\n')

    def _get_reference(self):
        return '!"{0}"'.format(_escape_string(self.string))
    _to_string = _get_reference

    def __eq__(self, other):
        if isinstance(other, MetaDataString):
            return self.string == other.string
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.string)