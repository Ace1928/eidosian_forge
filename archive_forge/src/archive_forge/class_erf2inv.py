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
class erf2inv(Function):
    """
    Two-argument Inverse error function. The erf2inv function is defined as:

    .. math ::
        \\mathrm{erf2}(x, w) = y \\quad \\Rightarrow \\quad \\mathrm{erf2inv}(x, y) = w

    Examples
    ========

    >>> from sympy import erf2inv, oo
    >>> from sympy.abc import x, y

    Several special values are known:

    >>> erf2inv(0, 0)
    0
    >>> erf2inv(1, 0)
    1
    >>> erf2inv(0, 1)
    oo
    >>> erf2inv(0, y)
    erfinv(y)
    >>> erf2inv(oo, y)
    erfcinv(-y)

    Differentiation with respect to $x$ and $y$ is supported:

    >>> from sympy import diff
    >>> diff(erf2inv(x, y), x)
    exp(-x**2 + erf2inv(x, y)**2)
    >>> diff(erf2inv(x, y), y)
    sqrt(pi)*exp(erf2inv(x, y)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erfcinv: Inverse complementary error function.

    References
    ==========

    .. [1] https://functions.wolfram.com/GammaBetaErf/InverseErf2/

    """

    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            return exp(self.func(x, y) ** 2 - x ** 2)
        elif argindex == 2:
            return sqrt(pi) * S.Half * exp(self.func(x, y) ** 2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y):
        if x is S.NaN or y is S.NaN:
            return S.NaN
        elif x.is_zero and y.is_zero:
            return S.Zero
        elif x.is_zero and y is S.One:
            return S.Infinity
        elif x is S.One and y.is_zero:
            return S.One
        elif x.is_zero:
            return erfinv(y)
        elif x is S.Infinity:
            return erfcinv(-y)
        elif y.is_zero:
            return x
        elif y is S.Infinity:
            return erfinv(x)
        if x.is_zero:
            if y.is_zero:
                return S.Zero
            else:
                return erfinv(y)
        if y.is_zero:
            return x

    def _eval_is_zero(self):
        x, y = self.args
        if x.is_zero and y.is_zero:
            return True