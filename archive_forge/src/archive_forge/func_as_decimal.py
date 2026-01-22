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
def as_decimal(self, prec):
    """Return a string representation of the algebraic number `self` in decimal notation
        using `prec` decimal places.

        >>> x = simplify(Sqrt(2))
        >>> x.as_decimal(10)
        '1.4142135623?'
        >>> x.as_decimal(20)
        '1.41421356237309504880?'
        """
    return Z3_get_numeral_decimal_string(self.ctx_ref(), self.as_ast(), prec)