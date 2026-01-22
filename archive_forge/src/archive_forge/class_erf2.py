from sympy.core import EulerGamma  # Must be imported from core, not core.numbers
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, expand_mul
from sympy.core.numbers import I, pi, Rational
from sympy.core.relational import is_eq
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, factorial2, RisingFactorial
from sympy.functions.elementary.complexes import  polar_lift, re, unpolarify
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt, root
from sympy.functions.elementary.exponential import exp, log, exp_polar
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin, sinc
from sympy.functions.special.hyper import hyper, meijerg
class erf2(Function):
    """
    Two-argument error function.

    Explanation
    ===========

    This function is defined as:

    .. math ::
        \\mathrm{erf2}(x, y) = \\frac{2}{\\sqrt{\\pi}} \\int_x^y e^{-t^2} \\mathrm{d}t

    Examples
    ========

    >>> from sympy import oo, erf2
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2(0, 0)
    0
    >>> erf2(x, x)
    0
    >>> erf2(x, oo)
    1 - erf(x)
    >>> erf2(x, -oo)
    -erf(x) - 1
    >>> erf2(oo, y)
    erf(y) - 1
    >>> erf2(-oo, y)
    erf(y) + 1

    In general one can pull out factors of -1:

    >>> erf2(-x, -y)
    -erf2(x, y)

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erf2(x, y))
    erf2(conjugate(x), conjugate(y))

    Differentiation with respect to $x$, $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2(x, y), x)
    -2*exp(-x**2)/sqrt(pi)
    >>> diff(erf2(x, y), y)
    2*exp(-y**2)/sqrt(pi)

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/Erf2/

    """

    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return -2 * exp(-x ** 2) / sqrt(pi)
        elif argindex == 2:
            return 2 * exp(-y ** 2) / sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        chk = (S.Infinity, S.NegativeInfinity, S.Zero)
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x == y:
            return S.Zero
        elif x in chk or y in chk:
            return erf(y) - erf(x)
        if isinstance(y, erf2inv) and y.args[0] == x:
            return y.args[1]
        if x.is_zero or y.is_zero or (x.is_extended_real and x.is_infinite) or (y.is_extended_real and y.is_infinite):
            return erf(y) - erf(x)
        sign_x = x.could_extract_minus_sign()
        sign_y = y.could_extract_minus_sign()
        if sign_x and sign_y:
            return -cls(-x, -y)
        elif sign_x or sign_y:
            return erf(y) - erf(x)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_rewrite_as_erf(self, x, y, **kwargs):
        return erf(y) - erf(x)

    def _eval_rewrite_as_erfc(self, x, y, **kwargs):
        return erfc(x) - erfc(y)

    def _eval_rewrite_as_erfi(self, x, y, **kwargs):
        return I * (erfi(I * x) - erfi(I * y))

    def _eval_rewrite_as_fresnels(self, x, y, **kwargs):
        return erf(y).rewrite(fresnels) - erf(x).rewrite(fresnels)

    def _eval_rewrite_as_fresnelc(self, x, y, **kwargs):
        return erf(y).rewrite(fresnelc) - erf(x).rewrite(fresnelc)

    def _eval_rewrite_as_meijerg(self, x, y, **kwargs):
        return erf(y).rewrite(meijerg) - erf(x).rewrite(meijerg)

    def _eval_rewrite_as_hyper(self, x, y, **kwargs):
        return erf(y).rewrite(hyper) - erf(x).rewrite(hyper)

    def _eval_rewrite_as_uppergamma(self, x, y, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return sqrt(y ** 2) / y * (S.One - uppergamma(S.Half, y ** 2) / sqrt(pi)) - sqrt(x ** 2) / x * (S.One - uppergamma(S.Half, x ** 2) / sqrt(pi))

    def _eval_rewrite_as_expint(self, x, y, **kwargs):
        return erf(y).rewrite(expint) - erf(x).rewrite(expint)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_is_zero(self):
        return is_eq(*self.args)