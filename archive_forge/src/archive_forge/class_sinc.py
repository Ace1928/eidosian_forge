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
class sinc(Function):
    """
    Represents an unnormalized sinc function:

    .. math::

        \\operatorname{sinc}(x) =
        \\begin{cases}
          \\frac{\\sin x}{x} & \\qquad x \\neq 0 \\\\
          1 & \\qquad x = 0
        \\end{cases}

    Examples
    ========

    >>> from sympy import sinc, oo, jn
    >>> from sympy.abc import x
    >>> sinc(x)
    sinc(x)

    * Automated Evaluation

    >>> sinc(0)
    1
    >>> sinc(oo)
    0

    * Differentiation

    >>> sinc(x).diff()
    cos(x)/x - sin(x)/x**2

    * Series Expansion

    >>> sinc(x).series()
    1 - x**2/6 + x**4/120 + O(x**6)

    * As zero'th order spherical Bessel Function

    >>> sinc(x).rewrite(jn)
    jn(0, x)

    See also
    ========

    sin

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Sinc_function

    """
    _singularities = (S.ComplexInfinity,)

    def fdiff(self, argindex=1):
        x = self.args[0]
        if argindex == 1:
            return cos(x) / x - sin(x) / x ** 2
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One
        if arg.is_Number:
            if arg in [S.Infinity, S.NegativeInfinity]:
                return S.Zero
            elif arg is S.NaN:
                return S.NaN
        if arg is S.ComplexInfinity:
            return S.NaN
        if arg.could_extract_minus_sign():
            return cls(-arg)
        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                if fuzzy_not(arg.is_zero):
                    return S.Zero
            elif (2 * pi_coeff).is_integer:
                return S.NegativeOne ** (pi_coeff - S.Half) / arg

    def _eval_nseries(self, x, n, logx, cdir=0):
        x = self.args[0]
        return (sin(x) / x)._eval_nseries(x, n, logx)

    def _eval_rewrite_as_jn(self, arg, **kwargs):
        from sympy.functions.special.bessel import jn
        return jn(0, arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return Piecewise((sin(arg) / arg, Ne(arg, S.Zero)), (S.One, S.true))

    def _eval_is_zero(self):
        if self.args[0].is_infinite:
            return True
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return fuzzy_and([pi_mult.is_integer, pi_mult.is_nonzero])
        if rest.is_Number and pi_mult.is_integer:
            return False

    def _eval_is_real(self):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return True
    _eval_is_finite = _eval_is_real