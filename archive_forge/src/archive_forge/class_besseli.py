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
class besseli(BesselBase):
    """
    Modified Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $I$ function is a solution to the modified Bessel equation

    .. math ::
        z^2 \\frac{\\mathrm{d}^2 w}{\\mathrm{d}z^2}
        + z \\frac{\\mathrm{d}w}{\\mathrm{d}z} + (z^2 + \\nu^2)^2 w = 0.

    It can be defined as

    .. math ::
        I_\\nu(z) = i^{-\\nu} J_\\nu(iz),

    where $J_\\nu(z)$ is the Bessel function of the first kind.

    Examples
    ========

    >>> from sympy import besseli
    >>> from sympy.abc import z, n
    >>> besseli(n, z).diff(z)
    besseli(n - 1, z)/2 + besseli(n + 1, z)/2

    See Also
    ========

    besselj, bessely, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/

    """
    _a = -S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif nu.is_integer and nu.is_zero is False or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and (not nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if im(z) in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z is S.Infinity:
            return S.Infinity
        if z is S.NegativeInfinity:
            return (-1) ** nu * S.Infinity
        if z.could_extract_minus_sign():
            return z ** nu * (-z) ** (-nu) * besseli(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besseli(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:
                return I ** (-nu) * besselj(nu, -newz)
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besseli(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2 * n * pi * nu * I) * besseli(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besseli(nnu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return exp(-I * pi * nu / 2) * besselj(nu, polar_lift(I) * z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return self._eval_rewrite_as_besselj(*self.args).rewrite(jn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)
        if e.is_positive:
            return arg ** nu / (2 ** nu * gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c * cdir ** e
            if not sign.is_negative:
                return exp(z) / sqrt(2 * pi * z)
            return self
        return super(besseli, self)._eval_as_leading_term(x, logx, cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self
        if exp.is_positive:
            newn = ceiling(n / exp)
            o = Order(x ** n, x)
            r = (z / 2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r ** 2) + o).removeO()
            term = r ** nu / gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1) // 2):
                term *= t / (k * (nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o
        return super(besseli, self)._eval_nseries(x, n, logx, cdir)