from __future__ import annotations
from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import ilcm, Float, Rational, _illegal
from sympy.core.power import integer_log, Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom
from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
from sympy.utilities.decorator import conserve_mpmath_dps
from mpmath import findroot
from sympy.solvers.polysys import solve_poly_system
from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product
import warnings
from sympy.solvers.bivariate import (
def _tsolve(eq, sym, **flags):
    """
    Helper for ``_solve`` that solves a transcendental equation with respect
    to the given symbol. Various equations containing powers and logarithms,
    can be solved.

    There is currently no guarantee that all solutions will be returned or
    that a real solution will be favored over a complex one.

    Either a list of potential solutions will be returned or None will be
    returned (in the case that no method was known to get a solution
    for the equation). All other errors (like the inability to cast an
    expression as a Poly) are unhandled.

    Examples
    ========

    >>> from sympy import log, ordered
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy.abc import x

    >>> list(ordered(tsolve(3**(2*x + 5) - 4, x)))
    [-5/2 + log(2)/log(3), (-5*log(3)/2 + log(2) + I*pi)/log(3)]

    >>> tsolve(log(x) + 2*x, x)
    [LambertW(2)/2]

    """
    if 'tsolve_saw' not in flags:
        flags['tsolve_saw'] = []
    if eq in flags['tsolve_saw']:
        return None
    else:
        flags['tsolve_saw'].append(eq)
    rhs, lhs = _invert(eq, sym)
    if lhs == sym:
        return [rhs]
    try:
        if lhs.is_Add:
            f = factor(powdenest(lhs - rhs))
            if f.is_Mul:
                return _vsolve(f, sym, **flags)
            if rhs:
                f = logcombine(lhs, force=flags.get('force', True))
                if f.count(log) != lhs.count(log):
                    if isinstance(f, log):
                        return _vsolve(f.args[0] - exp(rhs), sym, **flags)
                    return _tsolve(f - rhs, sym, **flags)
        elif lhs.is_Pow:
            if lhs.exp.is_Integer:
                if lhs - rhs != eq:
                    return _vsolve(lhs - rhs, sym, **flags)
            if sym not in lhs.exp.free_symbols:
                return _vsolve(lhs.base - rhs ** (1 / lhs.exp), sym, **flags)
            if any((t.is_Dummy for t in rhs.free_symbols)):
                raise NotImplementedError
            if not rhs:
                sol_base = _vsolve(lhs.base, sym, **flags)
                return [s for s in sol_base if lhs.exp.subs(sym, s) != 0]
            if not lhs.base.has(sym):
                if lhs.base == 0:
                    return _vsolve(lhs.exp, sym, **flags) if rhs != 0 else []
                if lhs.base == rhs.as_base_exp()[0]:
                    sol = _vsolve(lhs.exp - rhs.as_base_exp()[1], sym, **flags)
                else:
                    f = exp(log(lhs.base) * lhs.exp) - exp(log(rhs))
                    sol = _vsolve(f, sym, **flags)

                def equal(expr1, expr2):
                    _ = Dummy()
                    eq = checksol(expr1 - _, _, expr2)
                    if eq is None:
                        if nsimplify(expr1) != nsimplify(expr2):
                            return False
                        eq = expr1.equals(expr2)
                    return eq
                e_rat = nsimplify(log(abs(rhs)) / log(abs(lhs.base)))
                e_rat = simplify(posify(e_rat)[0])
                n, d = fraction(e_rat)
                if expand(lhs.base ** n - rhs ** d) == 0:
                    sol = [s for s in sol if not equal(lhs.exp.subs(sym, s), e_rat)]
                    sol.extend(_vsolve(lhs.exp - e_rat, sym, **flags))
                return list(set(sol))
            else:
                sol = []
                logform = lhs.exp * log(lhs.base) - log(rhs)
                if logform != lhs - rhs:
                    try:
                        sol.extend(_vsolve(logform, sym, **flags))
                    except NotImplementedError:
                        pass
                check = []
                if rhs == 1:
                    check.extend(_vsolve(lhs.exp, sym, **flags))
                    check.extend(_vsolve(lhs.base - 1, sym, **flags))
                    check.extend(_vsolve(lhs.base + 1, sym, **flags))
                elif rhs.is_Rational:
                    for d in (i for i in divisors(abs(rhs.p)) if i != 1):
                        e, t = integer_log(rhs.p, d)
                        if not t:
                            continue
                        for s in divisors(abs(rhs.q)):
                            if s ** e == rhs.q:
                                r = Rational(d, s)
                                check.extend(_vsolve(lhs.base - r, sym, **flags))
                                check.extend(_vsolve(lhs.base + r, sym, **flags))
                                check.extend(_vsolve(lhs.exp - e, sym, **flags))
                elif rhs.is_irrational:
                    b_l, e_l = lhs.base.as_base_exp()
                    n, d = (e_l * lhs.exp).as_numer_denom()
                    b, e = sqrtdenest(rhs).as_base_exp()
                    check = [sqrtdenest(i) for i in _vsolve(lhs.base - b, sym, **flags)]
                    check.extend([sqrtdenest(i) for i in _vsolve(lhs.exp - e, sym, **flags)])
                    if e_l * d != 1:
                        check.extend(_vsolve(b_l ** n - rhs ** (e_l * d), sym, **flags))
                for s in check:
                    ok = checksol(eq, sym, s)
                    if ok is None:
                        ok = eq.subs(sym, s).equals(0)
                    if ok:
                        sol.append(s)
                return list(set(sol))
        elif lhs.is_Function and len(lhs.args) == 1:
            if lhs.func in multi_inverses:
                soln = []
                for i in multi_inverses[type(lhs)](rhs):
                    soln.extend(_vsolve(lhs.args[0] - i, sym, **flags))
                return list(set(soln))
            elif lhs.func == LambertW:
                return _vsolve(lhs.args[0] - rhs * exp(rhs), sym, **flags)
        rewrite = lhs.rewrite(exp)
        if rewrite != lhs:
            return _vsolve(rewrite - rhs, sym, **flags)
    except NotImplementedError:
        pass
    if flags.pop('bivariate', True):
        logs = eq.atoms(log)
        spow = min({i.exp for j in logs for i in j.atoms(Pow) if i.base == sym} or {1})
        if spow != 1:
            p = sym ** spow
            u = Dummy('bivariate-cov')
            ueq = eq.subs(p, u)
            if not ueq.has_free(sym):
                sol = _vsolve(ueq, u, **flags)
                inv = _vsolve(p - u, sym)
                return [i.subs(u, s) for i in inv for s in sol]
        g = _filtered_gens(eq.as_poly(), sym)
        up_or_log = set()
        for gi in g:
            if isinstance(gi, (exp, log)) or (gi.is_Pow and gi.base == S.Exp1):
                up_or_log.add(gi)
            elif gi.is_Pow:
                gisimp = powdenest(expand_power_exp(gi))
                if gisimp.is_Pow and sym in gisimp.exp.free_symbols:
                    up_or_log.add(gi)
        eq_down = expand_log(expand_power_exp(eq)).subs(dict(list(zip(up_or_log, [0] * len(up_or_log)))))
        eq = expand_power_exp(factor(eq_down, deep=True) + (eq - eq_down))
        rhs, lhs = _invert(eq, sym)
        if lhs.has(sym):
            try:
                poly = lhs.as_poly()
                g = _filtered_gens(poly, sym)
                _eq = lhs - rhs
                sols = _solve_lambert(_eq, sym, g)
                for n, s in enumerate(sols):
                    ns = nsimplify(s)
                    if ns != s and ns.count_ops() <= s.count_ops():
                        ok = checksol(_eq, sym, ns)
                        if ok is None:
                            ok = _eq.subs(sym, ns).equals(0)
                        if ok:
                            sols[n] = ns
                return sols
            except NotImplementedError:
                if len(g) == 2:
                    try:
                        gpu = bivariate_type(lhs - rhs, *g)
                        if gpu is None:
                            raise NotImplementedError
                        g, p, u = gpu
                        flags['bivariate'] = False
                        inversion = _tsolve(g - u, sym, **flags)
                        if inversion:
                            sol = _vsolve(p, u, **flags)
                            return list({i.subs(u, s) for i in inversion for s in sol})
                    except NotImplementedError:
                        pass
                else:
                    pass
    if flags.pop('force', True):
        flags['force'] = False
        pos, reps = posify(lhs - rhs)
        if rhs == S.ComplexInfinity:
            return []
        for u, s in reps.items():
            if s == sym:
                break
        else:
            u = sym
        if pos.has(u):
            try:
                soln = _vsolve(pos, u, **flags)
                return [s.subs(reps) for s in soln]
            except NotImplementedError:
                pass
        else:
            pass
    return