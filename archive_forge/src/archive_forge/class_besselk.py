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
class besselk(BesselBase):
    """
    Modified Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $K$ function of order $\\nu$ is defined as

    .. math ::
        K_\\nu(z) = \\lim_{\\mu \\to \\nu} \\frac{\\pi}{2}
                   \\frac{I_{-\\mu}(z) -I_\\mu(z)}{\\sin(\\pi \\mu)},

    where $I_\\mu(z)$ is the modified Bessel function of the first kind.

    It is a solution of the modified Bessel equation, and linearly independent
    from $Y_\\nu$.

    Examples
    ========

    >>> from sympy import besselk
    >>> from sympy.abc import z, n
    >>> besselk(n, z).diff(z)
    -besselk(n - 1, z)/2 - besselk(n + 1, z)/2

    See Also
    ========

    besselj, besseli, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/

    """
    _a = S.One
    _b = -S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.Infinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, I * S.Infinity, I * S.NegativeInfinity):
            return S.Zero
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besselk(-nu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return pi * csc(pi * nu) * (besseli(-nu, z) - besseli(nu, z)) / 2

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        ai = self._eval_rewrite_as_besseli(*self.args)
        if ai:
            return ai.rewrite(besselj)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        ay = self._eval_rewrite_as_bessely(*self.args)
        if ay:
            return ay.rewrite(yn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        _, e = arg.as_coeff_exponent(x)
        if e.is_positive:
            term_one = (-1) ** (nu - 1) * log(z / 2) * besseli(nu, z)
            term_two = (z / 2) ** (-nu) * factorial(nu - 1) / 2 if nu.is_positive else S.Zero
            term_three = (-1) ** nu * (z / 2) ** nu / (2 * factorial(nu)) * (digamma(nu + 1) - S.EulerGamma)
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            return sqrt(pi) * exp(-z) / sqrt(2 * z)
        return super(besselk, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self
        if exp.is_positive and nu.is_integer:
            newn = ceiling(n / exp)
            bn = besseli(nu, z)
            a = ((-1) ** (nu - 1) * log(z / 2) * bn)._eval_nseries(x, n, logx, cdir)
            b, c = ([], [])
            o = Order(x ** n, x)
            r = (z / 2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r ** 2) + o).removeO()
            if nu > S.Zero:
                term = r ** (-nu) * factorial(nu - 1) / 2
                b.append(term)
                for k in range(1, nu):
                    denom = (k - nu) * k
                    if denom == S.Zero:
                        term *= t / k
                    else:
                        term *= t / denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)
            p = r ** nu * (-1) ** nu / (2 * factorial(nu))
            term = p * (digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1) // 2):
                p *= t / (k * (k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p * (digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            return a + Add(*b) + Add(*c)
        return super(besselk, self)._eval_nseries(x, n, logx, cdir)