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
def _nonlinear_2eq_order1_type5(func, t, eq):
    """
    Clairaut system of ODEs

    .. math:: x = t x' + F(x',y')

    .. math:: y = t y' + G(x',y')

    The following are solutions of the system

    `(i)` straight lines:

    .. math:: x = C_1 t + F(C_1, C_2), y = C_2 t + G(C_1, C_2)

    where `C_1` and `C_2` are arbitrary constants;

    `(ii)` envelopes of the above lines;

    `(iii)` continuously differentiable lines made up from segments of the lines
    `(i)` and `(ii)`.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    f = Wild('f')
    g = Wild('g')

    def check_type(x, y):
        r1 = eq[0].match(t * diff(x(t), t) - x(t) + f)
        r2 = eq[1].match(t * diff(y(t), t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t), t) - x(t) / t + f / t)
            r2 = eq[1].match(diff(y(t), t) - y(t) / t + g / t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t * diff(x(t), t) - x(t) + f)
            r2 = (-eq[1]).match(t * diff(y(t), t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t), t) - x(t) / t + f / t)
            r2 = (-eq[1]).match(diff(y(t), t) - y(t) / t + g / t)
        return [r1, r2]
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            [r1, r2] = check_type(x, y)
            if not (r1 and r2):
                [r1, r2] = check_type(y, x)
                x, y = (y, x)
    x1 = diff(x(t), t)
    y1 = diff(y(t), t)
    return {Eq(x(t), C1 * t + r1[f].subs(x1, C1).subs(y1, C2)), Eq(y(t), C2 * t + r2[g].subs(x1, C1).subs(y1, C2))}