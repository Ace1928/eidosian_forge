from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import ArgumentIndexError, expand_mul, Function
from sympy.core.numbers import pi, I, Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import bernoulli, factorial, genocchi, harmonic
from sympy.functions.elementary.complexes import re, unpolarify, Abs, polar_lift
from sympy.functions.elementary.exponential import log, exp_polar, exp
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polytools import Poly
class dirichlet_eta(Function):
    """
    Dirichlet eta function.

    Explanation
    ===========

    For $\\operatorname{Re}(s) > 0$ and $0 < x \\le 1$, this function is defined as

    .. math:: \\eta(s, a) = \\sum_{n=0}^\\infty \\frac{(-1)^n}{(n+a)^s}.

    It admits a unique analytic continuation to all of $\\mathbb{C}$ for any
    fixed $a$ not a nonpositive integer. It is an entire, unbranched function.

    It can be expressed using the Hurwitz zeta function as

    .. math:: \\eta(s, a) = \\zeta(s,a) - 2^{1-s} \\zeta\\left(s, \\frac{a+1}{2}\\right)

    and using the generalized Genocchi function as

    .. math:: \\eta(s, a) = \\frac{G(1-s, a)}{2(s-1)}.

    In both cases the limiting value of $\\log2 - \\psi(a) + \\psi\\left(\\frac{a+1}{2}\\right)$
    is used when $s = 1$.

    Examples
    ========

    >>> from sympy import dirichlet_eta, zeta
    >>> from sympy.abc import s
    >>> dirichlet_eta(s).rewrite(zeta)
    Piecewise((log(2), Eq(s, 1)), ((1 - 2**(1 - s))*zeta(s), True))

    See Also
    ========

    zeta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dirichlet_eta_function
    .. [2] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, s, a=None):
        if a is S.One:
            return cls(s)
        if a is None:
            if s == 1:
                return log(2)
            z = zeta(s)
            if not z.has(zeta):
                return (1 - 2 ** (1 - s)) * z
            return
        elif s == 1:
            from sympy.functions.special.gamma_functions import digamma
            return log(2) - digamma(a) + digamma((a + 1) / 2)
        z1 = zeta(s, a)
        z2 = zeta(s, (a + 1) / 2)
        if not z1.has(zeta) and (not z2.has(zeta)):
            return z1 - 2 ** (1 - s) * z2

    def _eval_rewrite_as_zeta(self, s, a=1, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        if a == 1:
            return Piecewise((log(2), Eq(s, 1)), ((1 - 2 ** (1 - s)) * zeta(s), True))
        return Piecewise((log(2) - digamma(a) + digamma((a + 1) / 2), Eq(s, 1)), (zeta(s, a) - 2 ** (1 - s) * zeta(s, (a + 1) / 2), True))

    def _eval_rewrite_as_genocchi(self, s, a=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((log(2) - digamma(a) + digamma((a + 1) / 2), Eq(s, 1)), (genocchi(1 - s, a) / (2 * (s - 1)), True))

    def _eval_evalf(self, prec):
        if all((i.is_number for i in self.args)):
            return self.rewrite(zeta)._eval_evalf(prec)