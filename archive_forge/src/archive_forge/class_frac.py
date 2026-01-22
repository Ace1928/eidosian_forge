from typing import Tuple as tTuple
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core import Add, S
from sympy.core.evalf import get_integer_part, PrecisionExhausted
from sympy.core.function import Function
from sympy.core.logic import fuzzy_or
from sympy.core.numbers import Integer
from sympy.core.relational import Gt, Lt, Ge, Le, Relational, is_eq
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import im, re
from sympy.multipledispatch import dispatch
class frac(Function):
    """Represents the fractional part of x

    For real numbers it is defined [1]_ as

    .. math::
        x - \\left\\lfloor{x}\\right\\rfloor

    Examples
    ========

    >>> from sympy import Symbol, frac, Rational, floor, I
    >>> frac(Rational(4, 3))
    1/3
    >>> frac(-Rational(4, 3))
    2/3

    returns zero for integer arguments

    >>> n = Symbol('n', integer=True)
    >>> frac(n)
    0

    rewrite as floor

    >>> x = Symbol('x')
    >>> frac(x).rewrite(floor)
    x - floor(x)

    for complex arguments

    >>> r = Symbol('r', real=True)
    >>> t = Symbol('t', real=True)
    >>> frac(t + I*r)
    I*frac(r) + frac(t)

    See Also
    ========

    sympy.functions.elementary.integers.floor
    sympy.functions.elementary.integers.ceiling

    References
    ===========

    .. [1] https://en.wikipedia.org/wiki/Fractional_part
    .. [2] https://mathworld.wolfram.com/FractionalPart.html

    """

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds

        def _eval(arg):
            if arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(0, 1)
            if arg.is_integer:
                return S.Zero
            if arg.is_number:
                if arg is S.NaN:
                    return S.NaN
                elif arg is S.ComplexInfinity:
                    return S.NaN
                else:
                    return arg - floor(arg)
            return cls(arg, evaluate=False)
        terms = Add.make_args(arg)
        real, imag = (S.Zero, S.Zero)
        for t in terms:
            if t.is_imaginary or (S.ImaginaryUnit * t).is_real:
                i = im(t)
                if not i.has(S.ImaginaryUnit):
                    imag += i
                else:
                    real += t
            else:
                real += t
        real = _eval(real)
        imag = _eval(imag)
        return real + S.ImaginaryUnit * imag

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    def _eval_is_finite(self):
        return True

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_integer

    def _eval_is_zero(self):
        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])

    def _eval_is_negative(self):
        return False

    def __ge__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            if other.is_extended_nonpositive:
                return S.true
            res = self._value_one_or_more(other)
            if res is not None:
                return not res
        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            res = self._value_one_or_more(other)
            if res is not None:
                return not res
            if other.is_extended_negative:
                return S.true
        return Gt(self, other, evaluate=False)

    def __le__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            if other.is_extended_negative:
                return S.false
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Le(self, other, evaluate=False)

    def __lt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            if other.is_extended_nonpositive:
                return S.false
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Lt(self, other, evaluate=False)

    def _value_one_or_more(self, other):
        if other.is_extended_real:
            if other.is_number:
                res = other >= 1
                if res and (not isinstance(res, Relational)):
                    return S.true
            if other.is_integer and other.is_positive:
                return S.true

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0.is_finite:
            if r.is_zero:
                ndir = arg.dir(x, cdir=cdir)
                if ndir.is_negative:
                    return S.One
                return (arg - arg0).as_leading_term(x, logx=logx, cdir=cdir)
            else:
                return r
        elif arg0 in (S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
            return AccumBounds(0, 1)
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1) + Order(x ** n, (x, 0))
            return o
        else:
            res = (arg - arg0)._eval_nseries(x, n, logx=logx, cdir=cdir)
            if r.is_zero:
                ndir = arg.dir(x, cdir=cdir)
                res += S.One if ndir.is_negative else S.Zero
            else:
                res += r
            return res