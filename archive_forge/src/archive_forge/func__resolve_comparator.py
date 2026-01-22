import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
def _resolve_comparator(self, set, args, kws):
    assert not kws
    arg, = args
    if arg == set:
        return signature(types.boolean, arg)