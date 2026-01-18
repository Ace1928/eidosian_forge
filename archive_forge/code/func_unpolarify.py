from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
def unpolarify(eq, subs=None, exponents_only=False):
    """
    If `p` denotes the projection from the Riemann surface of the logarithm to
    the complex line, return a simplified version `eq'` of `eq` such that
    `p(eq') = p(eq)`.
    Also apply the substitution subs in the end. (This is a convenience, since
    ``unpolarify``, in a certain sense, undoes :func:`polarify`.)

    Examples
    ========

    >>> from sympy import unpolarify, polar_lift, sin, I
    >>> unpolarify(polar_lift(I + 2))
    2 + I
    >>> unpolarify(sin(polar_lift(I + 7)))
    sin(7 + I)
    """
    if isinstance(eq, bool):
        return eq
    eq = sympify(eq)
    if subs is not None:
        return unpolarify(eq.subs(subs))
    changed = True
    pause = False
    if exponents_only:
        pause = True
    while changed:
        changed = False
        res = _unpolarify(eq, exponents_only, pause)
        if res != eq:
            changed = True
            eq = res
        if isinstance(res, bool):
            return res
    from sympy.functions.elementary.exponential import exp_polar
    return res.subs({exp_polar(0): 1, polar_lift(0): 0})