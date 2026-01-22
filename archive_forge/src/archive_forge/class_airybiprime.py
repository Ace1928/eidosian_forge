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
class airybiprime(AiryBase):
    """
    The derivative $\\operatorname{Bi}^\\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\\operatorname{Bi}^\\prime(z)$ is defined to be the
    function

    .. math::
        \\operatorname{Bi}^\\prime(z) := \\frac{\\mathrm{d} \\operatorname{Bi}(z)}{\\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybiprime
    >>> from sympy.abc import z

    >>> airybiprime(z)
    airybiprime(z)

    Several special values are known:

    >>> airybiprime(0)
    3**(1/6)/gamma(1/3)
    >>> from sympy import oo
    >>> airybiprime(oo)
    oo
    >>> airybiprime(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybiprime(z))
    airybiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybiprime(z), z)
    z*airybi(z)
    >>> diff(airybiprime(z), z, 2)
    z*airybiprime(z) + airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybiprime(z), z, 0, 3)
    3**(1/6)/gamma(1/3) + 3**(5/6)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybiprime(-2).evalf(50)
    0.27879516692116952268509756941098324140300059345163

    Rewrite $\\operatorname{Bi}^\\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybiprime(z).rewrite(hyper)
    3**(5/6)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) + 3**(1/6)*hyper((), (1/3,), z**3/9)/gamma(1/3)

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """
    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return 3 ** Rational(1, 6) / gamma(Rational(1, 3))
        if arg.is_zero:
            return 3 ** Rational(1, 6) / gamma(Rational(1, 3))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0] * airybi(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airybi(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = tt * Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return -z / sqrt(3) * (besselj(-tt, a) + besselj(tt, a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z / sqrt(3) * (besseli(-tt, a) + besseli(tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return sqrt(ot) * (b * besseli(-tt, tt * a) + z ** 2 * c * besseli(tt, tt * a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z ** 2 / (2 * root(3, 6) * gamma(Rational(2, 3)))
        pf2 = root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(5, 3)], z ** 3 / 9) + pf2 * hyper([], [Rational(1, 3)], z ** 3 / 9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols
        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild('c', exclude=[z])
            d = Wild('d', exclude=[z])
            m = Wild('m', exclude=[z])
            n = Wild('n', exclude=[z])
            M = arg.match(c * (d * z ** n) ** m)
            if M is not None:
                m = M[m]
                if (3 * m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = d ** m * z ** (n * m) / (d * z ** n) ** m
                    newarg = c * d ** m * z ** (n * m)
                    return S.Half * (sqrt(3) * (pf - S.One) * airyaiprime(newarg) + (pf + S.One) * airybiprime(newarg))