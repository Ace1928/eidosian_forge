import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_getattr
class NumberAttribute(AttributeTemplate):
    key = types.Number

    def resolve___class__(self, ty):
        return types.NumberClass(ty)

    def resolve_real(self, ty):
        return getattr(ty, 'underlying_float', ty)

    def resolve_imag(self, ty):
        return getattr(ty, 'underlying_float', ty)

    @bound_function('complex.conjugate')
    def resolve_conjugate(self, ty, args, kws):
        assert not args
        assert not kws
        return signature(ty)

    @bound_function('number.item')
    def resolve_item(self, ty, args, kws):
        assert not kws
        if not args:
            return signature(ty)