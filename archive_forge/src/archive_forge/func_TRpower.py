from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
def TRpower(rv):
    """Convert sin(x)**n and cos(x)**n with positive n to sums.

    Examples
    ========

    >>> from sympy.simplify.fu import TRpower
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> TRpower(sin(x)**6)
    -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 + 5/16
    >>> TRpower(sin(x)**3*cos(2*x)**4)
    (3*sin(x)/4 - sin(3*x)/4)*(cos(4*x)/2 + cos(8*x)/8 + 3/8)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Power-reduction_formulae

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, (sin, cos))):
            return rv
        b, n = rv.as_base_exp()
        x = b.args[0]
        if n.is_Integer and n.is_positive:
            if n.is_odd and isinstance(b, cos):
                rv = 2 ** (1 - n) * Add(*[binomial(n, k) * cos((n - 2 * k) * x) for k in range((n + 1) / 2)])
            elif n.is_odd and isinstance(b, sin):
                rv = 2 ** (1 - n) * S.NegativeOne ** ((n - 1) / 2) * Add(*[binomial(n, k) * S.NegativeOne ** k * sin((n - 2 * k) * x) for k in range((n + 1) / 2)])
            elif n.is_even and isinstance(b, cos):
                rv = 2 ** (1 - n) * Add(*[binomial(n, k) * cos((n - 2 * k) * x) for k in range(n / 2)])
            elif n.is_even and isinstance(b, sin):
                rv = 2 ** (1 - n) * S.NegativeOne ** (n / 2) * Add(*[binomial(n, k) * S.NegativeOne ** k * cos((n - 2 * k) * x) for k in range(n / 2)])
            if n.is_even:
                rv += 2 ** (-n) * binomial(n, n / 2)
        return rv
    return bottom_up(rv, f)