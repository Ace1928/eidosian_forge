import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
def _break_on_call_reduce(self, proto):
    raise TypeError('%r cannot be pickled' % self)