from typing import Tuple as tTuple, Union as tUnion
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols
class sec(ReciprocalTrigonometricFunction):
    """
    The secant function.

    Returns the secant of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import sec
    >>> from sympy.abc import x
    >>> sec(x**2).diff(x)
    2*x*tan(x**2)*sec(x**2)
    >>> sec(1).diff(x)
    0

    See Also
    ========

    sin, csc, cos, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sec

    """
    _reciprocal_of = cos
    _is_even = True

    def period(self, symbol=None):
        return self._period(symbol)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half_sq = cot(arg / 2) ** 2
        return (cot_half_sq + 1) / (cot_half_sq - 1)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / cos(arg)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg) / (cos(arg) * sin(arg))

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return 1 / cos(arg).rewrite(sin)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return 1 / cos(arg).rewrite(tan)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return csc(pi / 2 - arg, evaluate=False)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return tan(self.args[0]) * sec(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_complex and (arg / pi - S.Half).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2
            return S.NegativeOne ** k * euler(2 * k) / factorial(2 * k) * x ** (2 * k)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = (x0 + pi / 2) / pi
        if n.is_integer:
            lt = (arg - n * pi + pi / 2).as_leading_term(x)
            return S.NegativeOne ** n / lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self