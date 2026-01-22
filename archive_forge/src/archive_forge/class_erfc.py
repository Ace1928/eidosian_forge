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
class erfc(Function):
    """
    Complementary Error Function.

    Explanation
    ===========

    The function is defined as:

    .. math ::
        \\mathrm{erfc}(x) = \\frac{2}{\\sqrt{\\pi}} \\int_x^\\infty e^{-t^2} \\mathrm{d}t

    Examples
    ========

    >>> from sympy import I, oo, erfc
    >>> from sympy.abc import z

    Several special values are known:

    >>> erfc(0)
    1
    >>> erfc(oo)
    0
    >>> erfc(-oo)
    2
    >>> erfc(I*oo)
    -oo*I
    >>> erfc(-I*oo)
    oo*I

    The error function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(erfc(z))
    erfc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(erfc(z), z)
    -2*exp(-z**2)/sqrt(pi)

    It also follows

    >>> erfc(-z)
    2 - erfc(z)

    We can numerically evaluate the complementary error function to arbitrary
    precision on the whole complex plane:

    >>> erfc(4).evalf(30)
    0.0000000154172579002800188521596734869

    >>> erfc(4*I).evalf(30)
    1.0 - 1296959.73071763923152794095062*I

    See Also
    ========

    erf: Gaussian error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse Complementary error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/Erfc.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Erfc

    """
    unbranched = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -2 * exp(-self.args[0] ** 2) / sqrt(pi)
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfcinv

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg.is_zero:
                return S.One
        if isinstance(arg, erfinv):
            return S.One - arg.args[0]
        if isinstance(arg, erfcinv):
            return arg.args[0]
        if arg.is_zero:
            return S.One
        t = arg.extract_multiplicatively(I)
        if t in (S.Infinity, S.NegativeInfinity):
            return -arg
        if arg.could_extract_minus_sign():
            return 2 - cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return S.One
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = floor((n - 1) / S(2))
            if len(previous_terms) > 2:
                return -previous_terms[-2] * x ** 2 * (n - 2) / (n * k)
            else:
                return -2 * S.NegativeOne ** k * x ** n / (n * factorial(k) * sqrt(pi))

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return self.rewrite(erf).rewrite('tractable', deep=True, limitvar=limitvar)

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return S.One - erf(z)

    def _eval_rewrite_as_erfi(self, z, **kwargs):
        return S.One + I * erfi(I * z)

    def _eval_rewrite_as_fresnels(self, z, **kwargs):
        arg = (S.One - I) * z / sqrt(pi)
        return S.One - (S.One + I) * (fresnelc(arg) - I * fresnels(arg))

    def _eval_rewrite_as_fresnelc(self, z, **kwargs):
        arg = (S.One - I) * z / sqrt(pi)
        return S.One - (S.One + I) * (fresnelc(arg) - I * fresnels(arg))

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return S.One - z / sqrt(pi) * meijerg([S.Half], [], [0], [Rational(-1, 2)], z ** 2)

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return S.One - 2 * z / sqrt(pi) * hyper([S.Half], [3 * S.Half], -z ** 2)

    def _eval_rewrite_as_uppergamma(self, z, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return S.One - sqrt(z ** 2) / z * (S.One - uppergamma(S.Half, z ** 2) / sqrt(pi))

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return S.One - sqrt(z ** 2) / z + z * expint(S.Half, z ** 2) / sqrt(pi)

    def _eval_expand_func(self, **hints):
        return self.rewrite(erf)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if cdir == -1 else '+')
        if arg0.is_zero:
            return S.One
        else:
            return self.func(arg0)
    as_real_imag = real_to_real_as_real_imag

    def _eval_aseries(self, n, args0, x, logx):
        return S.One - erf(*self.args)._eval_aseries(n, args0, x, logx)