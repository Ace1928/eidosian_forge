import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
@alignstack.setter
def alignstack(self, val):
    assert val >= 0
    self._alignstack = val