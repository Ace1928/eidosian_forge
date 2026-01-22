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
class ArithRef(ExprRef):
    """Integer and Real expressions."""

    def sort(self):
        """Return the sort (type) of the arithmetical expression `self`.

        >>> Int('x').sort()
        Int
        >>> (Real('x') + 1).sort()
        Real
        """
        return ArithSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def is_int(self):
        """Return `True` if `self` is an integer expression.

        >>> x = Int('x')
        >>> x.is_int()
        True
        >>> (x + 1).is_int()
        True
        >>> y = Real('y')
        >>> (x + y).is_int()
        False
        """
        return self.sort().is_int()

    def is_real(self):
        """Return `True` if `self` is an real expression.

        >>> x = Real('x')
        >>> x.is_real()
        True
        >>> (x + 1).is_real()
        True
        """
        return self.sort().is_real()

    def __add__(self, other):
        """Create the Z3 expression `self + other`.

        >>> x = Int('x')
        >>> y = Int('y')
        >>> x + y
        x + y
        >>> (x + y).sort()
        Int
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(_mk_bin(Z3_mk_add, a, b), self.ctx)

    def __radd__(self, other):
        """Create the Z3 expression `other + self`.

        >>> x = Int('x')
        >>> 10 + x
        10 + x
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(_mk_bin(Z3_mk_add, b, a), self.ctx)

    def __mul__(self, other):
        """Create the Z3 expression `self * other`.

        >>> x = Real('x')
        >>> y = Real('y')
        >>> x * y
        x*y
        >>> (x * y).sort()
        Real
        """
        if isinstance(other, BoolRef):
            return If(other, self, 0)
        a, b = _coerce_exprs(self, other)
        return ArithRef(_mk_bin(Z3_mk_mul, a, b), self.ctx)

    def __rmul__(self, other):
        """Create the Z3 expression `other * self`.

        >>> x = Real('x')
        >>> 10 * x
        10*x
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(_mk_bin(Z3_mk_mul, b, a), self.ctx)

    def __sub__(self, other):
        """Create the Z3 expression `self - other`.

        >>> x = Int('x')
        >>> y = Int('y')
        >>> x - y
        x - y
        >>> (x - y).sort()
        Int
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(_mk_bin(Z3_mk_sub, a, b), self.ctx)

    def __rsub__(self, other):
        """Create the Z3 expression `other - self`.

        >>> x = Int('x')
        >>> 10 - x
        10 - x
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(_mk_bin(Z3_mk_sub, b, a), self.ctx)

    def __pow__(self, other):
        """Create the Z3 expression `self**other` (** is the power operator).

        >>> x = Real('x')
        >>> x**3
        x**3
        >>> (x**3).sort()
        Real
        >>> simplify(IntVal(2)**8)
        256
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(Z3_mk_power(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rpow__(self, other):
        """Create the Z3 expression `other**self` (** is the power operator).

        >>> x = Real('x')
        >>> 2**x
        2**x
        >>> (2**x).sort()
        Real
        >>> simplify(2**IntVal(8))
        256
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(Z3_mk_power(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __div__(self, other):
        """Create the Z3 expression `other/self`.

        >>> x = Int('x')
        >>> y = Int('y')
        >>> x/y
        x/y
        >>> (x/y).sort()
        Int
        >>> (x/y).sexpr()
        '(div x y)'
        >>> x = Real('x')
        >>> y = Real('y')
        >>> x/y
        x/y
        >>> (x/y).sort()
        Real
        >>> (x/y).sexpr()
        '(/ x y)'
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(Z3_mk_div(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __truediv__(self, other):
        """Create the Z3 expression `other/self`."""
        return self.__div__(other)

    def __rdiv__(self, other):
        """Create the Z3 expression `other/self`.

        >>> x = Int('x')
        >>> 10/x
        10/x
        >>> (10/x).sexpr()
        '(div 10 x)'
        >>> x = Real('x')
        >>> 10/x
        10/x
        >>> (10/x).sexpr()
        '(/ 10.0 x)'
        """
        a, b = _coerce_exprs(self, other)
        return ArithRef(Z3_mk_div(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __rtruediv__(self, other):
        """Create the Z3 expression `other/self`."""
        return self.__rdiv__(other)

    def __mod__(self, other):
        """Create the Z3 expression `other%self`.

        >>> x = Int('x')
        >>> y = Int('y')
        >>> x % y
        x%y
        >>> simplify(IntVal(10) % IntVal(3))
        1
        """
        a, b = _coerce_exprs(self, other)
        if z3_debug():
            _z3_assert(a.is_int(), 'Z3 integer expression expected')
        return ArithRef(Z3_mk_mod(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rmod__(self, other):
        """Create the Z3 expression `other%self`.

        >>> x = Int('x')
        >>> 10 % x
        10%x
        """
        a, b = _coerce_exprs(self, other)
        if z3_debug():
            _z3_assert(a.is_int(), 'Z3 integer expression expected')
        return ArithRef(Z3_mk_mod(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __neg__(self):
        """Return an expression representing `-self`.

        >>> x = Int('x')
        >>> -x
        -x
        >>> simplify(-(-x))
        x
        """
        return ArithRef(Z3_mk_unary_minus(self.ctx_ref(), self.as_ast()), self.ctx)

    def __pos__(self):
        """Return `self`.

        >>> x = Int('x')
        >>> +x
        x
        """
        return self

    def __le__(self, other):
        """Create the Z3 expression `other <= self`.

        >>> x, y = Ints('x y')
        >>> x <= y
        x <= y
        >>> y = Real('y')
        >>> x <= y
        ToReal(x) <= y
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_le(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __lt__(self, other):
        """Create the Z3 expression `other < self`.

        >>> x, y = Ints('x y')
        >>> x < y
        x < y
        >>> y = Real('y')
        >>> x < y
        ToReal(x) < y
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_lt(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __gt__(self, other):
        """Create the Z3 expression `other > self`.

        >>> x, y = Ints('x y')
        >>> x > y
        x > y
        >>> y = Real('y')
        >>> x > y
        ToReal(x) > y
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_gt(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __ge__(self, other):
        """Create the Z3 expression `other >= self`.

        >>> x, y = Ints('x y')
        >>> x >= y
        x >= y
        >>> y = Real('y')
        >>> x >= y
        ToReal(x) >= y
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_ge(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)