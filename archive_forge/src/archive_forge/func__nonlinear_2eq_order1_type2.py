from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve
from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from .single import SingleODEProblem, SingleODESolver, solver_map
def _nonlinear_2eq_order1_type2(x, y, t, eq):
    """
    Equations:

    .. math:: x' = e^{\\lambda x} F(x,y)

    .. math:: y' = g(y) F(x,y)

    Solution:

    .. math:: x = \\varphi(y), \\int \\frac{1}{g(y) F(\\varphi(y),y)} \\,dy = t + C_2

    where

    if `\\lambda \\neq 0`

    .. math:: \\varphi = -\\frac{1}{\\lambda} log(C_1 - \\lambda \\int \\frac{1}{g(y)} \\,dy)

    if `\\lambda = 0`

    .. math:: \\varphi = C_1 + \\int \\frac{1}{g(y)} \\,dy

    where `C_1` and `C_2` are arbitrary constants.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t), y(t)])
    f = Wild('f')
    u, v = symbols('u, v')
    r = eq[0].match(diff(x(t), t) - exp(n * x(t)) * f)
    g = ((diff(y(t), t) - eq[1]) / r[f]).subs(y(t), v)
    F = r[f].subs(x(t), u).subs(y(t), v)
    n = r[n]
    if n:
        phi = -1 / n * log(C1 - n * Integral(1 / g, v))
    else:
        phi = C1 + Integral(1 / g, v)
    phi = phi.doit()
    sol2 = solve(Integral(1 / (g * F.subs(u, phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t), phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol