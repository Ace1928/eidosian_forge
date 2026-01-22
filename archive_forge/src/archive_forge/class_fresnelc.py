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
class fresnelc(FresnelIntegral):
    """
    Fresnel integral C.

    Explanation
    ===========

    This function is defined by

    .. math:: \\operatorname{C}(z) = \\int_0^z \\cos{\\frac{\\pi}{2} t^2} \\mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import I, oo, fresnelc
    >>> from sympy.abc import z

    Several special values are known:

    >>> fresnelc(0)
    0
    >>> fresnelc(oo)
    1/2
    >>> fresnelc(-oo)
    -1/2
    >>> fresnelc(I*oo)
    I/2
    >>> fresnelc(-I*oo)
    -I/2

    In general one can pull out factors of -1 and $i$ from the argument:

    >>> fresnelc(-z)
    -fresnelc(z)
    >>> fresnelc(I*z)
    I*fresnelc(z)

    The Fresnel C integral obeys the mirror symmetry
    $\\overline{C(z)} = C(\\bar{z})$:

    >>> from sympy import conjugate
    >>> conjugate(fresnelc(z))
    fresnelc(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(fresnelc(z), z)
    cos(pi*z**2/2)

    Defining the Fresnel functions via an integral:

    >>> from sympy import integrate, pi, cos, expand_func
    >>> integrate(cos(pi*z**2/2), z)
    fresnelc(z)*gamma(1/4)/(4*gamma(5/4))
    >>> expand_func(integrate(cos(pi*z**2/2), z))
    fresnelc(z)

    We can numerically evaluate the Fresnel integral to arbitrary precision
    on the whole complex plane:

    >>> fresnelc(2).evalf(30)
    0.488253406075340754500223503357

    >>> fresnelc(-2*I).evalf(30)
    -0.488253406075340754500223503357*I

    See Also
    ========

    fresnels: Fresnel sine integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_integral
    .. [2] https://dlmf.nist.gov/7
    .. [3] https://mathworld.wolfram.com/FresnelIntegrals.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/FresnelC
    .. [5] The converging factors for the fresnel integrals
            by John W. Wrench Jr. and Vicki Alley

    """
    _trigfunc = cos
    _sign = S.One

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return -pi ** 2 * x ** 4 * (4 * n - 3) / (8 * n * (2 * n - 1) * (4 * n + 1)) * p
            else:
                return x * (-x ** 4) ** n * (S(2) ** (-2 * n) * pi ** (2 * n)) / ((4 * n + 1) * factorial(2 * n))

    def _eval_rewrite_as_erf(self, z, **kwargs):
        return (S.One - I) / 4 * (erf((S.One + I) / 2 * sqrt(pi) * z) + I * erf((S.One - I) / 2 * sqrt(pi) * z))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        return z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi ** 2 * z ** 4 / 16)

    def _eval_rewrite_as_meijerg(self, z, **kwargs):
        return pi * z ** Rational(3, 4) / (sqrt(2) * root(z ** 2, 4) * root(-z, 4)) * meijerg([], [1], [Rational(1, 4)], [Rational(3, 4), 0], -pi ** 2 * z ** 4 / 16)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.ComplexInfinity:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0 in [S.Infinity, S.NegativeInfinity]:
            s = 1 if arg0 is S.Infinity else -1
            return s * S.Half + Order(x, x)
        else:
            return self.func(arg0)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point in [S.Infinity, -S.Infinity]:
            z = self.args[0]
            p = [S.NegativeOne ** k * factorial(4 * k + 1) / (2 ** (2 * k + 2) * z ** (4 * k + 3) * 2 ** (2 * k) * factorial(2 * k)) for k in range(n) if 4 * k + 3 < n]
            q = [1 / (2 * z)] + [S.NegativeOne ** k * factorial(4 * k - 1) / (2 ** (2 * k + 1) * z ** (4 * k + 1) * 2 ** (2 * k - 1) * factorial(2 * k - 1)) for k in range(1, n) if 4 * k + 1 < n]
            p = [-sqrt(2 / pi) * t for t in p]
            q = [sqrt(2 / pi) * t for t in q]
            s = 1 if point is S.Infinity else -1
            return s * S.Half + (cos(z ** 2) * Add(*p) + sin(z ** 2) * Add(*q)).subs(x, sqrt(2 / pi) * x) + Order(1 / z ** n, x)
        return super()._eval_aseries(n, args0, x, logx)