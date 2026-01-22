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
class erfcinv(Function):
    """
    Inverse Complementary Error Function. The erfcinv function is defined as:

    .. math ::
        \\mathrm{erfc}(x) = y \\quad \\Rightarrow \\quad \\mathrm{erfcinv}(y) = x

    Examples
    ========

    >>> from sympy import erfcinv
    >>> from sympy.abc import x

    Several special values are known:

    >>> erfcinv(1)
    0
    >>> erfcinv(0)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(erfcinv(x), x)
    -sqrt(pi)*exp(erfcinv(x)**2)/2

    See Also
    ========

    erf: Gaussian error function.
    erfc: Complementary error function.
    erfi: Imaginary error function.
    erf2: Two-argument error function.
    erfinv: Inverse error function.
    erf2inv: Inverse two-argument error function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Error_function#Inverse_functions
    .. [2] https://functions.wolfram.com/GammaBetaErf/InverseErfc/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -sqrt(pi) * exp(self.func(self.args[0]) ** 2) * S.Half
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.

        """
        return erfc

    @classmethod
    def eval(cls, z):
        if z is S.NaN:
            return S.NaN
        elif z.is_zero:
            return S.Infinity
        elif z is S.One:
            return S.Zero
        elif z == 2:
            return S.NegativeInfinity
        if z.is_zero:
            return S.Infinity

    def _eval_rewrite_as_erfinv(self, z, **kwargs):
        return erfinv(1 - z)

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_infinite(self):
        return self.args[0].is_zero