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
class hn1(SphericalHankelBase):
    """
    Spherical Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\\nu^(1)(z) = j_\\nu(z) + i y_\\nu(z),

    where $j_\\nu(z)$ and $y_\\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(1)$ is calculated using the formula:

    .. math:: h_n^(1)(z) = j_{n}(z) + i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn1, hankel1, expand_func, yn, jn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn1(nu, z)))
    jn(nu, z) + I*yn(nu, z)
    >>> print(expand_func(hn1(0, z)))
    sin(z)/z - I*cos(z)/z
    >>> print(expand_func(hn1(1, z)))
    -I*sin(z)/z - cos(z)/z + sin(z)/z**2 - I*cos(z)/z**2
    >>> hn1(nu, z).rewrite(jn)
    (-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn1(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) + I*yn(nu, z)
    >>> hn1(nu, z).rewrite(hankel1)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel1(nu, z)/2

    See Also
    ========

    hn2, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    _hankel_kind_sign = S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel1(self, nu, z, **kwargs):
        return sqrt(pi / (2 * z)) * hankel1(nu, z)