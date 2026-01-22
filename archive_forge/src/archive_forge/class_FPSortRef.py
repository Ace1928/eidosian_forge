from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class FPSortRef(SortRef):
    """Floating-point sort."""

    def ebits(self):
        """Retrieves the number of bits reserved for the exponent in the FloatingPoint sort `self`.
        >>> b = FPSort(8, 24)
        >>> b.ebits()
        8
        """
        return int(Z3_fpa_get_ebits(self.ctx_ref(), self.ast))

    def sbits(self):
        """Retrieves the number of bits reserved for the significand in the FloatingPoint sort `self`.
        >>> b = FPSort(8, 24)
        >>> b.sbits()
        24
        """
        return int(Z3_fpa_get_sbits(self.ctx_ref(), self.ast))

    def cast(self, val):
        """Try to cast `val` as a floating-point expression.
        >>> b = FPSort(8, 24)
        >>> b.cast(1.0)
        1
        >>> b.cast(1.0).sexpr()
        '(fp #b0 #x7f #b00000000000000000000000)'
        """
        if is_expr(val):
            if z3_debug():
                _z3_assert(self.ctx == val.ctx, 'Context mismatch')
            return val
        else:
            return FPVal(val, None, self, self.ctx)