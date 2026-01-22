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
class atan2(InverseTrigonometricFunction):
    """
    The function ``atan2(y, x)`` computes `\\operatorname{atan}(y/x)` taking
    two arguments `y` and `x`.  Signs of both `y` and `x` are considered to
    determine the appropriate quadrant of `\\operatorname{atan}(y/x)`.
    The range is `(-\\pi, \\pi]`. The complete definition reads as follows:

    .. math::

        \\operatorname{atan2}(y, x) =
        \\begin{cases}
          \\arctan\\left(\\frac y x\\right) & \\qquad x > 0 \\\\
          \\arctan\\left(\\frac y x\\right) + \\pi& \\qquad y \\ge 0, x < 0 \\\\
          \\arctan\\left(\\frac y x\\right) - \\pi& \\qquad y < 0, x < 0 \\\\
          +\\frac{\\pi}{2} & \\qquad y > 0, x = 0 \\\\
          -\\frac{\\pi}{2} & \\qquad y < 0, x = 0 \\\\
          \\text{undefined} & \\qquad y = 0, x = 0
        \\end{cases}

    Attention: Note the role reversal of both arguments. The `y`-coordinate
    is the first argument and the `x`-coordinate the second.

    If either `x` or `y` is complex:

    .. math::

        \\operatorname{atan2}(y, x) =
            -i\\log\\left(\\frac{x + iy}{\\sqrt{x^2 + y^2}}\\right)

    Examples
    ========

    Going counter-clock wise around the origin we find the
    following angles:

    >>> from sympy import atan2
    >>> atan2(0, 1)
    0
    >>> atan2(1, 1)
    pi/4
    >>> atan2(1, 0)
    pi/2
    >>> atan2(1, -1)
    3*pi/4
    >>> atan2(0, -1)
    pi
    >>> atan2(-1, -1)
    -3*pi/4
    >>> atan2(-1, 0)
    -pi/2
    >>> atan2(-1, 1)
    -pi/4

    which are all correct. Compare this to the results of the ordinary
    `\\operatorname{atan}` function for the point `(x, y) = (-1, 1)`

    >>> from sympy import atan, S
    >>> atan(S(1)/-1)
    -pi/4
    >>> atan2(1, -1)
    3*pi/4

    where only the `\\operatorname{atan2}` function reurns what we expect.
    We can differentiate the function with respect to both arguments:

    >>> from sympy import diff
    >>> from sympy.abc import x, y
    >>> diff(atan2(y, x), x)
    -y/(x**2 + y**2)

    >>> diff(atan2(y, x), y)
    x/(x**2 + y**2)

    We can express the `\\operatorname{atan2}` function in terms of
    complex logarithms:

    >>> from sympy import log
    >>> atan2(y, x).rewrite(log)
    -I*log((x + I*y)/sqrt(x**2 + y**2))

    and in terms of `\\operatorname(atan)`:

    >>> from sympy import atan
    >>> atan2(y, x).rewrite(atan)
    Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)), (pi, re(x) < 0), (0, Ne(x, 0)), (nan, True))

    but note that this form is undefined on the negative real axis.

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://en.wikipedia.org/wiki/Atan2
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan2

    """

    @classmethod
    def eval(cls, y, x):
        from sympy.functions.special.delta_functions import Heaviside
        if x is S.NegativeInfinity:
            if y.is_zero:
                return pi
            return 2 * pi * Heaviside(re(y)) - pi
        elif x is S.Infinity:
            return S.Zero
        elif x.is_imaginary and y.is_imaginary and x.is_number and y.is_number:
            x = im(x)
            y = im(y)
        if x.is_extended_real and y.is_extended_real:
            if x.is_positive:
                return atan(y / x)
            elif x.is_negative:
                if y.is_negative:
                    return atan(y / x) - pi
                elif y.is_nonnegative:
                    return atan(y / x) + pi
            elif x.is_zero:
                if y.is_positive:
                    return pi / 2
                elif y.is_negative:
                    return -pi / 2
                elif y.is_zero:
                    return S.NaN
        if y.is_zero:
            if x.is_extended_nonzero:
                return pi * (S.One - Heaviside(x))
            if x.is_number:
                return Piecewise((pi, re(x) < 0), (0, Ne(x, 0)), (S.NaN, True))
        if x.is_number and y.is_number:
            return -S.ImaginaryUnit * log((x + S.ImaginaryUnit * y) / sqrt(x ** 2 + y ** 2))

    def _eval_rewrite_as_log(self, y, x, **kwargs):
        return -S.ImaginaryUnit * log((x + S.ImaginaryUnit * y) / sqrt(x ** 2 + y ** 2))

    def _eval_rewrite_as_atan(self, y, x, **kwargs):
        return Piecewise((2 * atan(y / (x + sqrt(x ** 2 + y ** 2))), Ne(y, 0)), (pi, re(x) < 0), (0, Ne(x, 0)), (S.NaN, True))

    def _eval_rewrite_as_arg(self, y, x, **kwargs):
        if x.is_extended_real and y.is_extended_real:
            return arg_f(x + y * S.ImaginaryUnit)
        n = x + S.ImaginaryUnit * y
        d = x ** 2 + y ** 2
        return arg_f(n / sqrt(d)) - S.ImaginaryUnit * log(abs(n) / sqrt(abs(d)))

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def fdiff(self, argindex):
        y, x = self.args
        if argindex == 1:
            return x / (x ** 2 + y ** 2)
        elif argindex == 2:
            return -y / (x ** 2 + y ** 2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        y, x = self.args
        if x.is_extended_real and y.is_extended_real:
            return super()._eval_evalf(prec)