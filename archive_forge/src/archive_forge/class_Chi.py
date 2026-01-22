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
class Chi(TrigonometricIntegral):
    """
    Cosh integral.

    Explanation
    ===========

    This function is defined for positive $x$ by

    .. math:: \\operatorname{Chi}(x) = \\gamma + \\log{x}
                         + \\int_0^x \\frac{\\cosh{t} - 1}{t} \\mathrm{d}t,

    where $\\gamma$ is the Euler-Mascheroni constant.

    We have

    .. math:: \\operatorname{Chi}(z) = \\operatorname{Ci}\\left(e^{i \\pi/2}z\\right)
                         - i\\frac{\\pi}{2},

    which holds for all polar $z$ and thus provides an analytic
    continuation to the Riemann surface of the logarithm.
    By lifting to the principal branch we obtain an analytic function on the
    cut complex plane.

    Examples
    ========

    >>> from sympy import Chi
    >>> from sympy.abc import z

    The $\\cosh$ integral is a primitive of $\\cosh(z)/z$:

    >>> Chi(z).diff(z)
    cosh(z)/z

    It has a logarithmic branch point at the origin:

    >>> from sympy import exp_polar, I, pi
    >>> Chi(z*exp_polar(2*I*pi))
    Chi(z) + 2*I*pi

    The $\\cosh$ integral behaves somewhat like ordinary $\\cosh$ under
    multiplication by $i$:

    >>> from sympy import polar_lift
    >>> Chi(polar_lift(I)*z)
    Ci(z) + I*pi/2
    >>> Chi(polar_lift(-1)*z)
    Chi(z) + I*pi

    It can also be expressed in terms of exponential integrals:

    >>> from sympy import expint
    >>> Chi(z).rewrite(expint)
    -expint(1, z)/2 - expint(1, z*exp_polar(I*pi))/2 - I*pi/2

    See Also
    ========

    Si: Sine integral.
    Ci: Cosine integral.
    Shi: Hyperbolic sine integral.
    Ei: Exponential integral.
    expint: Generalised exponential integral.
    E1: Special case of the generalised exponential integral.
    li: Logarithmic integral.
    Li: Offset logarithmic integral.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_integral

    """
    _trigfunc = cosh
    _atzero = S.ComplexInfinity

    @classmethod
    def _atinf(cls):
        return S.Infinity

    @classmethod
    def _atneginf(cls):
        return S.Infinity

    @classmethod
    def _minusfactor(cls, z):
        return Chi(z) + I * pi

    @classmethod
    def _Ifactor(cls, z, sign):
        return Ci(z) + I * pi / 2 * sign

    def _eval_rewrite_as_expint(self, z, **kwargs):
        return -I * pi / 2 - (E1(z) + E1(exp_polar(I * pi) * z)) / 2

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if arg0.is_zero:
            c, e = arg.as_coeff_exponent(x)
            logx = log(x) if logx is None else logx
            return log(c) + e * logx + EulerGamma
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self