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
class Si(TrigonometricIntegral):
    """
    Sine integral.

    Explanation
    ===========

    This function is defined by

    .. math:: \\operatorname{Si}(z) = \\int_0^z \\frac{\\sin{t}}{t} \\mathrm{d}t.

    It is an entire function.

    Examples
    ========

    >>> from sympy import Si
    >>> from sympy.abc import z

    The sine integral is an antiderivative of $sin(z)/z$:

    >>> Si(z).diff(z)
    sin(z)/z

    It is unbranched:

    >>> from sympy import exp_polar, I, pi
    >>> Si(z*exp_polar(2*I*pi))
    Si(z)

    Sine integral behaves much like ordinary sine under multiplication by ``I``:

    >>> Si(I*z)
    I*Shi(z)
    >>> Si(-z)
    -Si(z)

    It can also be expressed in terms of exponential integrals, but beware
    that the latter is branched:

    >>> from sympy import expint
    >>> Si(z).rewrite(expint)
    -I*(-expint(1, z*exp_polar(-I*pi/2))/2 +
         expint(1, z*exp_polar(I*pi/2))/2) + pi/2

    It can be rewritten in the form of sinc function (by definition):

    >>> from sympy import sinc
    >>> Si(z).rewrite(sinc)
    Integral(sinc(t), (t, 0, z))

    See Also
    ========

    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Chi: Hyperbolic cosine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    sinc: unnormalized sinc function
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """
    _trigfunc = sin
    _atzero = S.Zero

    @classmethod
    def _atinf(cls):
        return pi * S.Half

    @classmethod
    def _atneginf(cls):
        return -pi * S.Half

    @classmethod
    def _minusfactor(cls, z):
        return -Si(z)

    @classmethod
    def _Ifactor(cls, z, sign):
        return I * Shi(z) * sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return pi / 2 + (E1(polar_lift(I) * z) - E1(polar_lift(-I) * z)) / 2 / I

    def _eval_rewrite_as_sinc(self, z, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Symbol('t', Dummy=True)
        return Integral(sinc(t), (t, 0, z))

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        point = args0[0]
        if point is S.Infinity:
            z = self.args[0]
            p = [S.NegativeOne ** k * factorial(2 * k) / z ** (2 * k) for k in range(int((n - 1) / 2))] + [Order(1 / z ** n, x)]
            q = [S.NegativeOne ** k * factorial(2 * k + 1) / z ** (2 * k + 1) for k in range(int(n / 2) - 1)] + [Order(1 / z ** n, x)]
            return pi / 2 - cos(z) / z * Add(*p) - sin(z) / z * Add(*q)
        return super(Si, self)._eval_aseries(n, args0, x, logx)

    def _eval_is_zero(self):
        z = self.args[0]
        if z.is_zero:
            return True