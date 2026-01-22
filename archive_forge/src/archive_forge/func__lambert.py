from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq
def _lambert(eq, x):
    """
    Given an expression assumed to be in the form
        ``F(X, a..f) = a*log(b*X + c) + d*X + f = 0``
    where X = g(x) and x = g^-1(X), return the Lambert solution,
        ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``.
    """
    eq = _mexpand(expand_log(eq))
    mainlog = _mostfunc(eq, log, x)
    if not mainlog:
        return []
    other = eq.subs(mainlog, 0)
    if isinstance(-other, log):
        eq = (eq - other).subs(mainlog, mainlog.args[0])
        mainlog = mainlog.args[0]
        if not isinstance(mainlog, log):
            return []
        other = -(-other).args[0]
        eq += other
    if x not in other.free_symbols:
        return []
    d, f, X2 = _linab(other, x)
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    if a is None or x in a.free_symbols:
        return []
    logarg = mainlog.args[0]
    b, c, X1 = _linab(logarg, x)
    if X1 != X2:
        return []
    u = Dummy('rhs')
    xusolns = solve(X1 - u, x)
    lambert_real_branches = [-1, 0]
    sol = []
    num, den = ((c * d - b * f) / a / b).as_numer_denom()
    p, den = den.as_coeff_Mul()
    e = exp(num / den)
    t = Dummy('t')
    args = [d / (a * b) * t for t in roots(t ** p - e, t).keys()]
    for arg in args:
        for k in lambert_real_branches:
            w = LambertW(arg, k)
            if k and (not w.is_real):
                continue
            rhs = -c / b + a / d * w
            sol.extend((xu.subs(u, rhs) for xu in xusolns))
    return sol