import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
def _to_list(self, typ):
    attrs = super()._to_list(typ)
    if self.align:
        attrs.append('align {0:d}'.format(self.align))
    if self.dereferenceable:
        attrs.append('dereferenceable({0:d})'.format(self.dereferenceable))
    if self.dereferenceable_or_null:
        dref = 'dereferenceable_or_null({0:d})'
        attrs.append(dref.format(self.dereferenceable_or_null))
    return attrs