from __future__ import annotations
from functools import reduce
from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.ntheory import sieve
from sympy.polys.polytools import Poly
from math import factorial as _factorial, prod, sqrt as _sqrt
class FallingFactorial(CombinatorialFunction):
    """
    Falling factorial (related to rising factorial) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by

    .. math:: \\texttt{ff(x, k)} = (x)_k = x \\cdot (x-1) \\cdots (x-k+1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or [1]_.

    When `x` is a `~.Poly` instance of degree $\\ge 1$ with single variable,
    `(x)_k = x(y) \\cdot x(y-1) \\cdots x(y-k+1)`, where `y` is the
    variable of `x`. This is as described in

    >>> from sympy import ff, Poly, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer=True)

    >>> ff(x, 0)
    1
    >>> ff(5, 5)
    120
    >>> ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)
    True
    >>> ff(Poly(x**2, x), 2)
    Poly(x**4 - 2*x**3 + x**2, x, domain='ZZ')
    >>> ff(n, n)
    factorial(n)

    Rewriting is complicated unless the relationship between
    the arguments is known, but falling factorial can
    be rewritten in terms of gamma, factorial and binomial
    and rising factorial.

    >>> from sympy import factorial, rf, gamma, binomial, Symbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> F = ff(n, n - 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  F.rewrite(i)
    ...
    RisingFactorial(3, n - 2)
    FallingFactorial(n, n - 2)
    factorial(n)/2
    binomial(n, n - 2)*factorial(n - 2)
    gamma(n + 1)/2

    See Also
    ========

    factorial, factorial2, RisingFactorial

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FallingFactorial.html
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    def eval(cls, x, k):
        x = sympify(x)
        k = sympify(k)
        if x is S.NaN or k is S.NaN:
            return S.NaN
        elif k.is_integer and x == k:
            return factorial(x)
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            elif k.is_positive:
                if x is S.Infinity:
                    return S.Infinity
                elif x is S.NegativeInfinity:
                    if k.is_odd:
                        return S.NegativeInfinity
                    else:
                        return S.Infinity
                elif isinstance(x, Poly):
                    gens = x.gens
                    if len(gens) != 1:
                        raise ValueError('ff only defined for polynomials on one generator')
                    else:
                        return reduce(lambda r, i: r * x.shift(-i), range(int(k)), 1)
                else:
                    return reduce(lambda r, i: r * (x - i), range(int(k)), 1)
            elif x is S.Infinity:
                return S.Infinity
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif isinstance(x, Poly):
                gens = x.gens
                if len(gens) != 1:
                    raise ValueError('rf only defined for polynomials on one generator')
                else:
                    return 1 / reduce(lambda r, i: r * x.shift(i), range(1, abs(int(k)) + 1), 1)
            else:
                return 1 / reduce(lambda r, i: r * (x + i), range(1, abs(int(k)) + 1), 1)

    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        if not piecewise:
            if (x < 0) == True:
                return S.NegativeOne ** k * gamma(k - x) / gamma(-x)
            return gamma(x + 1) / gamma(x - k + 1)
        return Piecewise((gamma(x + 1) / gamma(x - k + 1), x >= 0), (S.NegativeOne ** k * gamma(k - x) / gamma(-x), True))

    def _eval_rewrite_as_RisingFactorial(self, x, k, **kwargs):
        return rf(x - k + 1, k)

    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        if k.is_integer:
            return factorial(k) * binomial(x, k)

    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        if x.is_integer and k.is_integer:
            return Piecewise((factorial(x) / factorial(-k + x), x >= 0), (S.NegativeOne ** k * factorial(k - x - 1) / factorial(-x - 1), True))

    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        if limitvar:
            k_lim = k.subs(limitvar, S.Infinity)
            if k_lim is S.Infinity:
                return S.NegativeOne ** k * gamma(k - x).rewrite('tractable', deep=True) / gamma(-x)
            elif k_lim is S.NegativeInfinity:
                return gamma(x + 1) / gamma(x - k + 1).rewrite('tractable', deep=True)
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    def _eval_is_integer(self):
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer, self.args[1].is_nonnegative))