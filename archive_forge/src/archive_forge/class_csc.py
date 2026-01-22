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
class csc(ReciprocalTrigonometricFunction):
    """
    The cosecant function.

    Returns the cosecant of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import csc
    >>> from sympy.abc import x
    >>> csc(x**2).diff(x)
    -2*x*cot(x**2)*csc(x**2)
    >>> csc(1).diff(x)
    0

    See Also
    ========

    sin, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Csc

    """
    _reciprocal_of = sin
    _is_odd = True

    def period(self, symbol=None):
        return self._period(symbol)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return 1 / sin(arg)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg) / (sin(arg) * cos(arg))

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(arg / 2)
        return (1 + cot_half ** 2) / (2 * cot_half)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / sin(arg).rewrite(cos)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(pi / 2 - arg, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return 1 / sin(arg).rewrite(tan)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -cot(self.args[0]) * csc(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg / pi).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2 + 1
            return S.NegativeOne ** (k - 1) * 2 * (2 ** (2 * k - 1) - 1) * bernoulli(2 * k) * x ** (2 * k - 1) / factorial(2 * k)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0 / pi
        if n.is_integer:
            lt = (arg - n * pi).as_leading_term(x)
            return S.NegativeOne ** n / lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self