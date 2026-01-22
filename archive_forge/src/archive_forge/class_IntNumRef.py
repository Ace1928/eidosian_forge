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
class IntNumRef(ArithRef):
    """Integer values."""

    def as_long(self):
        """Return a Z3 integer numeral as a Python long (bignum) numeral.

        >>> v = IntVal(1)
        >>> v + 1
        1 + 1
        >>> v.as_long() + 1
        2
        """
        if z3_debug():
            _z3_assert(self.is_int(), 'Integer value expected')
        return int(self.as_string())

    def as_string(self):
        """Return a Z3 integer numeral as a Python string.
        >>> v = IntVal(100)
        >>> v.as_string()
        '100'
        """
        return Z3_get_numeral_string(self.ctx_ref(), self.as_ast())

    def as_binary_string(self):
        """Return a Z3 integer numeral as a Python binary string.
        >>> v = IntVal(10)
        >>> v.as_binary_string()
        '1010'
        """
        return Z3_get_numeral_binary_string(self.ctx_ref(), self.as_ast())