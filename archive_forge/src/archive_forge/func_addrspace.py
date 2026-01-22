import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
@property
def addrspace(self):
    if not isinstance(self.type, types.PointerType):
        raise TypeError('Only pointer constant have address spaces')
    return self.type.addrspace