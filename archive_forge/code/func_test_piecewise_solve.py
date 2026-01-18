from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.function import (Function, diff, expand)
from sympy.core.mul import Mul
from sympy.core.mod import Mod
from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import (Piecewise,
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, ITE, Not, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.printing import srepr
from sympy.sets.contains import Contains
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.utilities.lambdify import lambdify
def test_piecewise_solve():
    abs2 = Piecewise((-x, x <= 0), (x, x > 0))
    f = abs2.subs(x, x - 2)
    assert solve(f, x) == [2]
    assert solve(f - 1, x) == [1, 3]
    f = Piecewise(((x - 2) ** 2, x >= 0), (1, True))
    assert solve(f, x) == [2]
    g = Piecewise(((x - 5) ** 5, x >= 4), (f, True))
    assert solve(g, x) == [2, 5]
    g = Piecewise(((x - 5) ** 5, x >= 4), (f, x < 4))
    assert solve(g, x) == [2, 5]
    g = Piecewise(((x - 5) ** 5, x >= 2), (f, x < 2))
    assert solve(g, x) == [5]
    g = Piecewise(((x - 5) ** 5, x >= 2), (f, True))
    assert solve(g, x) == [5]
    g = Piecewise(((x - 5) ** 5, x >= 2), (f, True), (10, False))
    assert solve(g, x) == [5]
    g = Piecewise(((x - 5) ** 5, x >= 2), (-x + 2, x - 2 <= 0), (x - 2, x - 2 > 0))
    assert solve(g, x) == [5]
    assert solve(Piecewise((x - 2, x > 2), (2 - x, True)) - 3) == [-1, 5]
    f = Piecewise(((x - 2) ** 2, x >= 0), (0, True))
    raises(NotImplementedError, lambda: solve(f, x))

    def nona(ans):
        return list(filter(lambda x: x is not S.NaN, ans))
    p = Piecewise((x ** 2 - 4, x < y), (x - 2, True))
    ans = solve(p, x)
    assert nona([i.subs(y, -2) for i in ans]) == [2]
    assert nona([i.subs(y, 2) for i in ans]) == [-2, 2]
    assert nona([i.subs(y, 3) for i in ans]) == [-2, 2]
    assert ans == [Piecewise((-2, y > -2), (S.NaN, True)), Piecewise((2, y <= 2), (S.NaN, True)), Piecewise((2, y > 2), (S.NaN, True))]
    absxm3 = Piecewise((x - 3, 0 <= x - 3), (3 - x, 0 > x - 3))
    assert solve(absxm3 - y, x) == [Piecewise((-y + 3, -y < 0), (S.NaN, True)), Piecewise((y + 3, y >= 0), (S.NaN, True))]
    p = Symbol('p', positive=True)
    assert solve(absxm3 - p, x) == [-p + 3, p + 3]
    f = Function('f')
    assert solve(Eq(-f(x), Piecewise((1, x > 0), (0, True))), f(x)) == [Piecewise((-1, x > 0), (0, True))]
    f = Piecewise((2 * x ** 2, And(0 < x, x < 1)), (2, True))
    assert solve(f - 1) == [1 / sqrt(2)]