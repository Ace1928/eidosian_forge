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
class BoolRef(ExprRef):
    """All Boolean expressions are instances of this class."""

    def sort(self):
        return BoolSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def __add__(self, other):
        if isinstance(other, BoolRef):
            other = If(other, 1, 0)
        return If(self, 1, 0) + other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        """Create the Z3 expression `self * other`.
        """
        if isinstance(other, int) and other == 1:
            return If(self, 1, 0)
        if isinstance(other, int) and other == 0:
            return IntVal(0, self.ctx)
        if isinstance(other, BoolRef):
            other = If(other, 1, 0)
        return If(self, other, 0)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __xor__(self, other):
        return Xor(self, other)

    def __invert__(self):
        return Not(self)