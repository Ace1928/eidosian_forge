from functools import wraps
from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn
from mpmath import mp, workprec
class yn(SphericalBesselBase):
    """
    Spherical Bessel function of the second kind.

    Explanation
    ===========

    This function is another solution to the spherical Bessel equation, and
    linearly independent from $j_n$. It can be defined as

    .. math ::
        y_\\nu(z) = \\sqrt{\\frac{\\pi}{2z}} Y_{\\nu + \\frac{1}{2}}(z),

    where $Y_\\nu(z)$ is the Bessel function of the second kind.

    For integral orders $n$, $y_n$ is calculated using the formula:

    .. math:: y_n(z) = (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, yn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(yn(0, z)))
    -cos(z)/z
    >>> expand_func(yn(1, z)) == -cos(z)/z**2-sin(z)/z
    True
    >>> yn(nu, z).rewrite(besselj)
    (-1)**(nu + 1)*sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(-nu - 1/2, z)/2
    >>> yn(nu, z).rewrite(bessely)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(nu + 1/2, z)/2
    >>> yn(2, 5.2+0.3j).evalf(20)
    0.18525034196069722536 + 0.014895573969924817587*I

    See Also
    ========

    besselj, bessely, besselk, jn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return S.NegativeOne ** (nu + 1) * sqrt(pi / (2 * z)) * besselj(-nu - S.Half, z)

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return sqrt(pi / (2 * z)) * bessely(nu + S.Half, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return S.NegativeOne ** (nu + 1) * jn(-nu - 1, z)

    def _expand(self, **hints):
        return _yn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(bessely)._eval_evalf(prec)