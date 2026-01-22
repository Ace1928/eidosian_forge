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
def _nonlinear_3eq_order1_type1(x, y, z, t, eq):
    """
    Equations:

    .. math:: a x' = (b - c) y z, \\enspace b y' = (c - a) z x, \\enspace c z' = (a - b) x y

    First Integrals:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and
    `z` and on substituting the resulting expressions into the first equation of the
    system, we arrives at a separable first-order equation on `x`. Similarly doing that
    for other two equations, we will arrive at first order equation on `y` and `z` too.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0401.pdf

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    r = (diff(x(t), t) - eq[0]).match(p * y(t) * z(t))
    r.update((diff(y(t), t) - eq[1]).match(q * z(t) * x(t)))
    r.update((diff(z(t), t) - eq[2]).match(s * x(t) * y(t)))
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()
    val = solve([n1 * u - d1 * v + d1 * w, d2 * u + n2 * v - d2 * w, d3 * u - d3 * v - n3 * w], [u, v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    b = vals[0].subs(w, c)
    a = vals[1].subs(w, c)
    y_x = sqrt((c * C1 - C2 - a * (c - a) * x(t) ** 2) / (b * (c - b)))
    z_x = sqrt((b * C1 - C2 - a * (b - a) * x(t) ** 2) / (c * (b - c)))
    z_y = sqrt((a * C1 - C2 - b * (a - b) * y(t) ** 2) / (c * (a - c)))
    x_y = sqrt((c * C1 - C2 - b * (c - b) * y(t) ** 2) / (a * (c - a)))
    x_z = sqrt((b * C1 - C2 - c * (b - c) * z(t) ** 2) / (a * (b - a)))
    y_z = sqrt((a * C1 - C2 - c * (a - c) * z(t) ** 2) / (b * (a - b)))
    sol1 = dsolve(a * diff(x(t), t) - (b - c) * y_x * z_x)
    sol2 = dsolve(b * diff(y(t), t) - (c - a) * z_y * x_y)
    sol3 = dsolve(c * diff(z(t), t) - (a - b) * x_z * y_z)
    return [sol1, sol2, sol3]