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
class airyaiprime(AiryBase):
    """
    The derivative $\\operatorname{Ai}^\\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\\operatorname{Ai}^\\prime(z)$ is defined to be the
    function

    .. math::
        \\operatorname{Ai}^\\prime(z) := \\frac{\\mathrm{d} \\operatorname{Ai}(z)}{\\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyaiprime
    >>> from sympy.abc import z

    >>> airyaiprime(z)
    airyaiprime(z)

    Several special values are known:

    >>> airyaiprime(0)
    -3**(2/3)/(3*gamma(1/3))
    >>> from sympy import oo
    >>> airyaiprime(oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyaiprime(z))
    airyaiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyaiprime(z), z)
    z*airyai(z)
    >>> diff(airyaiprime(z), z, 2)
    z*airyaiprime(z) + airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyaiprime(z), z, 0, 3)
    -3**(2/3)/(3*gamma(1/3)) + 3**(1/3)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyaiprime(-2).evalf(50)
    0.61825902074169104140626429133247528291577794512415

    Rewrite $\\operatorname{Ai}^\\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyaiprime(z).rewrite(hyper)
    3**(1/3)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) - 3**(2/3)*hyper((), (1/3,), z**3/9)/(3*gamma(1/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airybiprime: Derivative of the Airy function of the second kind.

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
                return S.Zero
        if arg.is_zero:
            return S.NegativeOne / (3 ** Rational(1, 3) * gamma(Rational(1, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0] * airyai(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airyai(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return z / 3 * (besselj(-tt, tt * a) - besselj(tt, tt * a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z / 3 * (besseli(tt, a) - besseli(-tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return ot * (z ** 2 * c * besseli(tt, tt * a) - b * besseli(-ot, tt * a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z ** 2 / (2 * 3 ** Rational(2, 3) * gamma(Rational(2, 3)))
        pf2 = 1 / (root(3, 3) * gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(5, 3)], z ** 3 / 9) - pf2 * hyper([], [Rational(1, 3)], z ** 3 / 9)

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
                    return S.Half * ((pf + S.One) * airyaiprime(newarg) + (pf - S.One) / sqrt(3) * airybiprime(newarg))