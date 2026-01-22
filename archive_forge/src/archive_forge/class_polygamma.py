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
class polygamma(Function):
    """
    The function ``polygamma(n, z)`` returns ``log(gamma(z)).diff(n + 1)``.

    Explanation
    ===========

    It is a meromorphic function on $\\mathbb{C}$ and defined as the $(n+1)$-th
    derivative of the logarithm of the gamma function:

    .. math::
        \\psi^{(n)} (z) := \\frac{\\mathrm{d}^{n+1}}{\\mathrm{d} z^{n+1}} \\log\\Gamma(z).

    For `n` not a nonnegative integer the generalization by Espinosa and Moll [5]_
    is used:

    .. math:: \\psi(s,z) = \\frac{\\zeta'(s+1, z) + (\\gamma + \\psi(-s)) \\zeta(s+1, z)}
        {\\Gamma(-s)}

    Examples
    ========

    Several special values are known:

    >>> from sympy import S, polygamma
    >>> polygamma(0, 1)
    -EulerGamma
    >>> polygamma(0, 1/S(2))
    -2*log(2) - EulerGamma
    >>> polygamma(0, 1/S(3))
    -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))
    >>> polygamma(0, 1/S(4))
    -pi/2 - log(4) - log(2) - EulerGamma
    >>> polygamma(0, 2)
    1 - EulerGamma
    >>> polygamma(0, 23)
    19093197/5173168 - EulerGamma

    >>> from sympy import oo, I
    >>> polygamma(0, oo)
    oo
    >>> polygamma(0, -oo)
    oo
    >>> polygamma(0, I*oo)
    oo
    >>> polygamma(0, -I*oo)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import Symbol, diff
    >>> x = Symbol("x")
    >>> diff(polygamma(0, x), x)
    polygamma(1, x)
    >>> diff(polygamma(0, x), x, 2)
    polygamma(2, x)
    >>> diff(polygamma(0, x), x, 3)
    polygamma(3, x)
    >>> diff(polygamma(1, x), x)
    polygamma(2, x)
    >>> diff(polygamma(1, x), x, 2)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x, 2)
    polygamma(4, x)

    >>> n = Symbol("n")
    >>> diff(polygamma(n, x), x)
    polygamma(n + 1, x)
    >>> diff(polygamma(n, x), x, 2)
    polygamma(n + 2, x)

    We can rewrite ``polygamma`` functions in terms of harmonic numbers:

    >>> from sympy import harmonic
    >>> polygamma(0, x).rewrite(harmonic)
    harmonic(x - 1) - EulerGamma
    >>> polygamma(2, x).rewrite(harmonic)
    2*harmonic(x - 1, 3) - 2*zeta(3)
    >>> ni = Symbol("n", integer=True)
    >>> polygamma(ni, x).rewrite(harmonic)
    (-1)**(n + 1)*(-harmonic(x - 1, n + 1) + zeta(n + 1))*factorial(n)

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Polygamma_function
    .. [2] https://mathworld.wolfram.com/PolygammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma/
    .. [4] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/
    .. [5] O. Espinosa and V. Moll, "A generalized polygamma function",
           *Integral Transforms and Special Functions* (2004), 101-115.

    """

    @classmethod
    def eval(cls, n, z):
        if n is S.NaN or z is S.NaN:
            return S.NaN
        elif z is oo:
            return oo if n.is_zero else S.Zero
        elif z.is_Integer and z.is_nonpositive:
            return S.ComplexInfinity
        elif n is S.NegativeOne:
            return loggamma(z) - log(2 * pi) / 2
        elif n.is_zero:
            if z is -oo or z.extract_multiplicatively(I) in (oo, -oo):
                return oo
            elif z.is_Integer:
                return harmonic(z - 1) - S.EulerGamma
            elif z.is_Rational:
                p, q = z.as_numer_denom()
                if q <= 6:
                    return expand_func(polygamma(S.Zero, z, evaluate=False))
        elif n.is_integer and n.is_nonnegative:
            nz = unpolarify(z)
            if z != nz:
                return polygamma(n, nz)
            if z.is_Integer:
                return S.NegativeOne ** (n + 1) * factorial(n) * zeta(n + 1, z)
            elif z is S.Half:
                return S.NegativeOne ** (n + 1) * factorial(n) * (2 ** (n + 1) - 1) * zeta(n + 1)

    def _eval_is_real(self):
        if self.args[0].is_positive and self.args[1].is_positive:
            return True

    def _eval_is_complex(self):
        z = self.args[1]
        is_negative_integer = fuzzy_and([z.is_negative, z.is_integer])
        return fuzzy_and([z.is_complex, fuzzy_not(is_negative_integer)])

    def _eval_is_positive(self):
        n, z = self.args
        if n.is_positive:
            if n.is_odd and z.is_real:
                return True
            if n.is_even and z.is_positive:
                return False

    def _eval_is_negative(self):
        n, z = self.args
        if n.is_positive:
            if n.is_even and z.is_positive:
                return True
            if n.is_odd and z.is_real:
                return False

    def _eval_expand_func(self, **hints):
        n, z = self.args
        if n.is_Integer and n.is_nonnegative:
            if z.is_Add:
                coeff = z.args[0]
                if coeff.is_Integer:
                    e = -(n + 1)
                    if coeff > 0:
                        tail = Add(*[Pow(z - i, e) for i in range(1, int(coeff) + 1)])
                    else:
                        tail = -Add(*[Pow(z + i, e) for i in range(int(-coeff))])
                    return polygamma(n, z - coeff) + S.NegativeOne ** n * factorial(n) * tail
            elif z.is_Mul:
                coeff, z = z.as_two_terms()
                if coeff.is_Integer and coeff.is_positive:
                    tail = [polygamma(n, z + Rational(i, coeff)) for i in range(int(coeff))]
                    if n == 0:
                        return Add(*tail) / coeff + log(coeff)
                    else:
                        return Add(*tail) / coeff ** (n + 1)
                z *= coeff
        if n == 0 and z.is_Rational:
            p, q = z.as_numer_denom()
            part_1 = -S.EulerGamma - pi * cot(p * pi / q) / 2 - log(q) + Add(*[cos(2 * k * pi * p / q) * log(2 * sin(k * pi / q)) for k in range(1, q)])
            if z > 0:
                n = floor(z)
                z0 = z - n
                return part_1 + Add(*[1 / (z0 + k) for k in range(n)])
            elif z < 0:
                n = floor(1 - z)
                z0 = z + n
                return part_1 - Add(*[1 / (z0 - 1 - k) for k in range(n)])
        if n == -1:
            return loggamma(z) - log(2 * pi) / 2
        if n.is_integer is False or n.is_nonnegative is False:
            s = Dummy('s')
            dzt = zeta(s, z).diff(s).subs(s, n + 1)
            return (dzt + (S.EulerGamma + digamma(-n)) * zeta(n + 1, z)) / gamma(-n)
        return polygamma(n, z)

    def _eval_rewrite_as_zeta(self, n, z, **kwargs):
        if n.is_integer and n.is_positive:
            return S.NegativeOne ** (n + 1) * factorial(n) * zeta(n + 1, z)

    def _eval_rewrite_as_harmonic(self, n, z, **kwargs):
        if n.is_integer:
            if n.is_zero:
                return harmonic(z - 1) - S.EulerGamma
            else:
                return S.NegativeOne ** (n + 1) * factorial(n) * (zeta(n + 1) - harmonic(z - 1, n + 1))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        from sympy.series.order import Order
        n, z = [a.as_leading_term(x) for a in self.args]
        o = Order(z, x)
        if n == 0 and o.contains(1 / x):
            logx = log(x) if logx is None else logx
            return o.getn() * logx
        else:
            return self.func(n, z)

    def fdiff(self, argindex=2):
        if argindex == 2:
            n, z = self.args[:2]
            return polygamma(n + 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[1] != oo or not (self.args[0].is_Integer and self.args[0].is_nonnegative):
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[1]
        N = self.args[0]
        if N == 0:
            r = log(z) - 1 / (2 * z)
            o = None
            if n < 2:
                o = Order(1 / z, x)
            else:
                m = ceiling((n + 1) // 2)
                l = [bernoulli(2 * k) / (2 * k * z ** (2 * k)) for k in range(1, m)]
                r -= Add(*l)
                o = Order(1 / z ** n, x)
            return r._eval_nseries(x, n, logx) + o
        else:
            fac = gamma(N)
            e0 = fac + N * fac / (2 * z)
            m = ceiling((n + 1) // 2)
            for k in range(1, m):
                fac = fac * (2 * k + N - 1) * (2 * k + N - 2) / (2 * k * (2 * k - 1))
                e0 += bernoulli(2 * k) * fac / z ** (2 * k)
            o = Order(1 / z ** (2 * m), x)
            if n == 0:
                o = Order(1 / z, x)
            elif n == 1:
                o = Order(1 / z ** 2, x)
            r = e0._eval_nseries(z, n, logx) + o
            return (-1 * (-1 / z) ** N * r)._eval_nseries(x, n, logx)

    def _eval_evalf(self, prec):
        if not all((i.is_number for i in self.args)):
            return
        s = self.args[0]._to_mpmath(prec + 12)
        z = self.args[1]._to_mpmath(prec + 12)
        if mp.isint(z) and z <= 0:
            return S.ComplexInfinity
        with workprec(prec + 12):
            if mp.isint(s) and s >= 0:
                res = mp.polygamma(s, z)
            else:
                zt = mp.zeta(s + 1, z)
                dzt = mp.zeta(s + 1, z, 1)
                res = (dzt + (mp.euler + mp.digamma(-s)) * zt) * mp.rgamma(-s)
        return Expr._from_mpmath(res, prec)