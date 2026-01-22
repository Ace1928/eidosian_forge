from math import prod
from sympy.core import Add, S, Dummy, expand_func
from sympy.core.expr import Expr
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and, fuzzy_not
from sympy.core.numbers import Rational, pi, oo, I
from sympy.core.power import Pow
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
class loggamma(Function):
    """
    The ``loggamma`` function implements the logarithm of the
    gamma function (i.e., $\\log\\Gamma(x)$).

    Examples
    ========

    Several special values are known. For numerical integral
    arguments we have:

    >>> from sympy import loggamma
    >>> loggamma(-2)
    oo
    >>> loggamma(0)
    oo
    >>> loggamma(1)
    0
    >>> loggamma(2)
    0
    >>> loggamma(3)
    log(2)

    And for symbolic values:

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> loggamma(n)
    log(gamma(n))
    >>> loggamma(-n)
    oo

    For half-integral values:

    >>> from sympy import S
    >>> loggamma(S(5)/2)
    log(3*sqrt(pi)/4)
    >>> loggamma(n/2)
    log(2**(1 - n)*sqrt(pi)*gamma(n)/gamma(n/2 + 1/2))

    And general rational arguments:

    >>> from sympy import expand_func
    >>> L = loggamma(S(16)/3)
    >>> expand_func(L).doit()
    -5*log(3) + loggamma(1/3) + log(4) + log(7) + log(10) + log(13)
    >>> L = loggamma(S(19)/4)
    >>> expand_func(L).doit()
    -4*log(4) + loggamma(3/4) + log(3) + log(7) + log(11) + log(15)
    >>> L = loggamma(S(23)/7)
    >>> expand_func(L).doit()
    -3*log(7) + log(2) + loggamma(2/7) + log(9) + log(16)

    The ``loggamma`` function has the following limits towards infinity:

    >>> from sympy import oo
    >>> loggamma(oo)
    oo
    >>> loggamma(-oo)
    zoo

    The ``loggamma`` function obeys the mirror symmetry
    if $x \\in \\mathbb{C} \\setminus \\{-\\infty, 0\\}$:

    >>> from sympy.abc import x
    >>> from sympy import conjugate
    >>> conjugate(loggamma(x))
    loggamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(loggamma(x), x)
    polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(loggamma(x), x, 0, 4).cancel()
    -log(x) - EulerGamma*x + pi**2*x**2/12 - x**3*zeta(3)/3 + O(x**4)

    We can numerically evaluate the ``loggamma`` function
    to arbitrary precision on the whole complex plane:

    >>> from sympy import I
    >>> loggamma(5).evalf(30)
    3.17805383034794561964694160130
    >>> loggamma(I).evalf(20)
    -0.65092319930185633889 - 1.8724366472624298171*I

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/LogGammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/LogGamma/

    """

    @classmethod
    def eval(cls, z):
        if z.is_integer:
            if z.is_nonpositive:
                return oo
            elif z.is_positive:
                return log(gamma(z))
        elif z.is_rational:
            p, q = z.as_numer_denom()
            if p.is_positive and q == 2:
                return log(sqrt(pi) * 2 ** (1 - p) * gamma(p) / gamma((p + 1) * S.Half))
        if z is oo:
            return oo
        elif abs(z) is oo:
            return S.ComplexInfinity
        if z is S.NaN:
            return S.NaN

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        z = self.args[0]
        if z.is_Rational:
            p, q = z.as_numer_denom()
            n = p // q
            p = p - n * q
            if p.is_positive and q.is_positive and (p < q):
                k = Dummy('k')
                if n.is_positive:
                    return loggamma(p / q) - n * log(q) + Sum(log((k - 1) * q + p), (k, 1, n))
                elif n.is_negative:
                    return loggamma(p / q) - n * log(q) + pi * I * n - Sum(log(k * q - p), (k, 1, -n))
                elif n.is_zero:
                    return loggamma(p / q)
        return self

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] != oo:
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[0]
        r = log(z) * (z - S.Half) - z + log(2 * pi) / 2
        l = [bernoulli(2 * k) / (2 * k * (2 * k - 1) * z ** (2 * k - 1)) for k in range(1, n)]
        o = None
        if n == 0:
            o = Order(1, x)
        else:
            o = Order(1 / z ** n, x)
        return (r + Add(*l))._eval_nseries(x, n, logx) + o

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return log(gamma(z))

    def _eval_is_real(self):
        z = self.args[0]
        if z.is_positive:
            return True
        elif z.is_nonpositive:
            return False

    def _eval_conjugate(self):
        z = self.args[0]
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(z.conjugate())

    def fdiff(self, argindex=1):
        if argindex == 1:
            return polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)