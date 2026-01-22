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
class SphericalHankelBase(SphericalBesselBase):

    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return sqrt(pi / (2 * z)) * (besselj(nu + S.Half, z) + hks * I * S.NegativeOne ** (nu + 1) * besselj(-nu - S.Half, z))

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return sqrt(pi / (2 * z)) * (S.NegativeOne ** nu * bessely(-nu - S.Half, z) + hks * I * bessely(nu + S.Half, z))

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z).rewrite(yn) + hks * I * yn(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z) + hks * I * yn(nu, z).rewrite(jn)

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        else:
            nu = self.order
            z = self.argument
            hks = self._hankel_kind_sign
            return jn(nu, z) + hks * I * yn(nu, z)

    def _expand(self, **hints):
        n = self.order
        z = self.argument
        hks = self._hankel_kind_sign
        return (_jn(n, z) + hks * I * _yn(n, z)).expand()

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)