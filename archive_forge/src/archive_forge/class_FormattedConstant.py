import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class FormattedConstant(Constant):
    """
    A constant with an already formatted IR representation.
    """

    def __init__(self, typ, constant):
        assert isinstance(constant, str)
        Constant.__init__(self, typ, constant)

    def _to_string(self):
        return self.constant

    def _get_reference(self):
        return self.constant