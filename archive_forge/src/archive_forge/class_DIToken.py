import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class DIToken:
    """
    A debug information enumeration value that should appear bare in
    the emitted metadata.

    Use this to wrap known constants, e.g. the DW_* enumerations.
    """

    def __init__(self, value):
        self.value = value