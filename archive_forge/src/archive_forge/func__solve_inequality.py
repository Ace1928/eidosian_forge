import itertools
from sympy.calculus.util import (continuous_domain, periodicity,
from sympy.core import Symbol, Dummy, sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Eq, Ge, Lt
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import im, Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent
def _solve_inequality(ie, s, linear=False):
    """Return the inequality with s isolated on the left, if possible.
    If the relationship is non-linear, a solution involving And or Or
    may be returned. False or True are returned if the relationship
    is never True or always True, respectively.

    If `linear` is True (default is False) an `s`-dependent expression
    will be isolated on the left, if possible
    but it will not be solved for `s` unless the expression is linear
    in `s`. Furthermore, only "safe" operations which do not change the
    sense of the relationship are applied: no division by an unsigned
    value is attempted unless the relationship involves Eq or Ne and
    no division by a value not known to be nonzero is ever attempted.

    Examples
    ========

    >>> from sympy import Eq, Symbol
    >>> from sympy.solvers.inequalities import _solve_inequality as f
    >>> from sympy.abc import x, y

    For linear expressions, the symbol can be isolated:

    >>> f(x - 2 < 0, x)
    x < 2
    >>> f(-x - 6 < x, x)
    x > -3

    Sometimes nonlinear relationships will be False

    >>> f(x**2 + 4 < 0, x)
    False

    Or they may involve more than one region of values:

    >>> f(x**2 - 4 < 0, x)
    (-2 < x) & (x < 2)

    To restrict the solution to a relational, set linear=True
    and only the x-dependent portion will be isolated on the left:

    >>> f(x**2 - 4 < 0, x, linear=True)
    x**2 < 4

    Division of only nonzero quantities is allowed, so x cannot
    be isolated by dividing by y:

    >>> y.is_nonzero is None  # it is unknown whether it is 0 or not
    True
    >>> f(x*y < 1, x)
    x*y < 1

    And while an equality (or inequality) still holds after dividing by a
    non-zero quantity

    >>> nz = Symbol('nz', nonzero=True)
    >>> f(Eq(x*nz, 1), x)
    Eq(x, 1/nz)

    the sign must be known for other inequalities involving > or <:

    >>> f(x*nz <= 1, x)
    nz*x <= 1
    >>> p = Symbol('p', positive=True)
    >>> f(x*p <= 1, x)
    x <= 1/p

    When there are denominators in the original expression that
    are removed by expansion, conditions for them will be returned
    as part of the result:

    >>> f(x < x*(2/x - 1), x)
    (x < 1) & Ne(x, 0)
    """
    from sympy.solvers.solvers import denoms
    if s not in ie.free_symbols:
        return ie
    if ie.rhs == s:
        ie = ie.reversed
    if ie.lhs == s and s not in ie.rhs.free_symbols:
        return ie

    def classify(ie, s, i):
        try:
            v = ie.subs(s, i)
            if v is S.NaN:
                return v
            elif v not in (True, False):
                return
            return v
        except TypeError:
            return S.NaN
    rv = None
    oo = S.Infinity
    expr = ie.lhs - ie.rhs
    try:
        p = Poly(expr, s)
        if p.degree() == 0:
            rv = ie.func(p.as_expr(), 0)
        elif not linear and p.degree() > 1:
            raise NotImplementedError
    except (PolynomialError, NotImplementedError):
        if not linear:
            try:
                rv = reduce_rational_inequalities([[ie]], s)
            except PolynomialError:
                rv = solve_univariate_inequality(ie, s)
            okoo = classify(ie, s, oo)
            if okoo is S.true and classify(rv, s, oo) is S.false:
                rv = rv.subs(s < oo, True)
            oknoo = classify(ie, s, -oo)
            if oknoo is S.true and classify(rv, s, -oo) is S.false:
                rv = rv.subs(-oo < s, True)
                rv = rv.subs(s > -oo, True)
            if rv is S.true:
                rv = s <= oo if okoo is S.true else s < oo
                if oknoo is not S.true:
                    rv = And(-oo < s, rv)
        else:
            p = Poly(expr)
    conds = []
    if rv is None:
        e = p.as_expr()
        rhs = 0
        b, ax = e.as_independent(s, as_Add=True)
        e -= b
        rhs -= b
        ef = factor_terms(e)
        a, e = ef.as_independent(s, as_Add=False)
        if a.is_zero != False or (a.is_negative == a.is_positive is None and ie.rel_op not in ('!=', '==')):
            e = ef
            a = S.One
        rhs /= a
        if a.is_positive:
            rv = ie.func(e, rhs)
        else:
            rv = ie.reversed.func(e, rhs)
        beginning_denoms = denoms(ie.lhs) | denoms(ie.rhs)
        current_denoms = denoms(rv)
        for d in beginning_denoms - current_denoms:
            c = _solve_inequality(Eq(d, 0), s, linear=linear)
            if isinstance(c, Eq) and c.lhs == s:
                if classify(rv, s, c.rhs) is S.true:
                    conds.append(~c)
        for i in (-oo, oo):
            if classify(rv, s, i) is S.true and classify(ie, s, i) is not S.true:
                conds.append(s < i if i is oo else i < s)
    conds.append(rv)
    return And(*conds)