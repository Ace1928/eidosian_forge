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
def _nonlinear_3eq_order1_type4(x, y, z, t, eq):
    """
    Equations:

    .. math:: x' = c z F_2 - b y F_3, \\enspace y' = a x F_3 - c z F_1, \\enspace z' = b y F_1 - a x F_2

    where `F_n = F_n (x, y, z, t)`

    1. First integral:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    where `C` is an arbitrary constant.

    2. Assuming the function `F_n` is independent of `t`: `F_n = F_n (x, y, z)`. Then on
    eliminating `t` and `z` from the first two equations of the system, one arrives at
    the first-order equation

    .. math:: \\frac{dy}{dx} = \\frac{a x F_3 (x, y, z) - c z F_1 (x, y, z)}
                {c z F_2 (x, y, z) - b y F_3 (x, y, z)}

    where `z = \\pm \\sqrt{\\frac{1}{c} (C_1 - a x^{2} - b y^{2})}`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0405.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t), t) - z(t) * F2 + y(t) * F3)
    r = collect_const(r1[F2]).match(s * F2)
    r.update(collect_const(r1[F3]).match(q * F3))
    if eq[1].has(r[F2]) and (not eq[1].has(r[F3])):
        r[F2], r[F3] = (r[F3], r[F2])
        r[s], r[q] = (-r[q], -r[s])
    r.update((diff(y(t), t) - eq[1]).match(p * x(t) * r[F3] - r[s] * z(t) * F1))
    a = r[p]
    b = r[q]
    c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    x_yz = sqrt((C1 - b * v ** 2 - c * w ** 2) / a)
    y_zx = sqrt((C1 - c * w ** 2 - a * u ** 2) / b)
    z_xy = sqrt((C1 - a * u ** 2 - b * v ** 2) / c)
    y_x = dsolve(diff(v(u), u) - ((a * u * F3 - c * w * F1) / (c * w * F2 - b * v * F3)).subs(w, z_xy).subs(v, v(u))).rhs
    z_x = dsolve(diff(w(u), u) - ((b * v * F1 - a * u * F2) / (c * w * F2 - b * v * F3)).subs(v, y_zx).subs(w, w(u))).rhs
    z_y = dsolve(diff(w(v), v) - ((b * v * F1 - a * u * F2) / (a * u * F3 - c * w * F1)).subs(u, x_yz).subs(w, w(v))).rhs
    x_y = dsolve(diff(u(v), v) - ((c * w * F2 - b * v * F3) / (a * u * F3 - c * w * F1)).subs(w, z_xy).subs(u, u(v))).rhs
    y_z = dsolve(diff(v(w), w) - ((a * u * F3 - c * w * F1) / (b * v * F1 - a * u * F2)).subs(u, x_yz).subs(v, v(w))).rhs
    x_z = dsolve(diff(u(w), w) - ((c * w * F2 - b * v * F3) / (b * v * F1 - a * u * F2)).subs(v, y_zx).subs(u, u(w))).rhs
    sol1 = dsolve(diff(u(t), t) - (c * w * F2 - b * v * F3).subs(v, y_x).subs(w, z_x).subs(u, u(t))).rhs
    sol2 = dsolve(diff(v(t), t) - (a * u * F3 - c * w * F1).subs(u, x_y).subs(w, z_y).subs(v, v(t))).rhs
    sol3 = dsolve(diff(w(t), t) - (b * v * F1 - a * u * F2).subs(u, x_z).subs(v, y_z).subs(w, w(t))).rhs
    return [sol1, sol2, sol3]