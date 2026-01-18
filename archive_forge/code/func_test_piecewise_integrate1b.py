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
def test_piecewise_integrate1b():
    g = Piecewise((1, x > 0), (0, Eq(x, 0)), (-1, x < 0))
    assert integrate(g, (x, -1, 1)) == 0
    g = Piecewise((1, x - y < 0), (0, True))
    assert integrate(g, (y, -oo, 0)) == -Min(0, x)
    assert g.subs(x, -3).integrate((y, -oo, 0)) == 3
    assert integrate(g, (y, 0, -oo)) == Min(0, x)
    assert integrate(g, (y, 0, oo)) == -Max(0, x) + oo
    assert integrate(g, (y, -oo, 42)) == -Min(42, x) + 42
    assert integrate(g, (y, -oo, oo)) == -x + oo
    g = Piecewise((0, x < 0), (x, x <= 1), (1, True))
    gy1 = g.integrate((x, y, 1))
    g1y = g.integrate((x, 1, y))
    for yy in (-1, S.Half, 2):
        assert g.integrate((x, yy, 1)) == gy1.subs(y, yy)
        assert g.integrate((x, 1, yy)) == g1y.subs(y, yy)
    assert gy1 == Piecewise((-Min(1, Max(0, y)) ** 2 / 2 + S.Half, y < 1), (-y + 1, True))
    assert g1y == Piecewise((Min(1, Max(0, y)) ** 2 / 2 - S.Half, y < 1), (y - 1, True))