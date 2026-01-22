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
class factorial2(CombinatorialFunction):
    """The double factorial `n!!`, not to be confused with `(n!)!`

    The double factorial is defined for nonnegative integers and for odd
    negative integers as:

    .. math:: n!! = \\begin{cases} 1 & n = 0 \\\\
                    n(n-2)(n-4) \\cdots 1 & n\\ \\text{positive odd} \\\\
                    n(n-2)(n-4) \\cdots 2 & n\\ \\text{positive even} \\\\
                    (n+2)!!/(n+2) & n\\ \\text{negative odd} \\end{cases}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Double_factorial

    Examples
    ========

    >>> from sympy import factorial2, var
    >>> n = var('n')
    >>> n
    n
    >>> factorial2(n + 1)
    factorial2(n + 1)
    >>> factorial2(5)
    15
    >>> factorial2(-1)
    1
    >>> factorial2(-5)
    1/3

    See Also
    ========

    factorial, RisingFactorial, FallingFactorial
    """

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if not arg.is_Integer:
                raise ValueError('argument must be nonnegative integer or negative odd integer')
            if arg.is_nonnegative:
                if arg.is_even:
                    k = arg / 2
                    return 2 ** k * factorial(k)
                return factorial(arg) / factorial2(arg - 1)
            if arg.is_odd:
                return arg * S.NegativeOne ** ((1 - arg) / 2) / factorial2(-arg)
            raise ValueError('argument must be nonnegative integer or negative odd integer')

    def _eval_is_even(self):
        n = self.args[0]
        if n.is_integer:
            if n.is_odd:
                return False
            if n.is_even:
                if n.is_positive:
                    return True
                if n.is_zero:
                    return False

    def _eval_is_integer(self):
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return (n + 3).is_nonnegative

    def _eval_is_odd(self):
        n = self.args[0]
        if n.is_odd:
            return (n + 3).is_nonnegative
        if n.is_even:
            if n.is_positive:
                return False
            if n.is_zero:
                return True

    def _eval_is_positive(self):
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return ((n + 1) / 2).is_even

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        return 2 ** (n / 2) * gamma(n / 2 + 1) * Piecewise((1, Eq(Mod(n, 2), 0)), (sqrt(2 / pi), Eq(Mod(n, 2), 1)))