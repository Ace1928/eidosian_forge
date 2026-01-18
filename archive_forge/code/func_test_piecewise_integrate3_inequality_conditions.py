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
def test_piecewise_integrate3_inequality_conditions():
    from sympy.utilities.iterables import cartes
    lim = (x, 0, 5)
    N = (-2, -1, 0, 1, 2, 5, 6, 7)
    p = Piecewise((1, x > a), (2, x > b), (0, True))
    ans = p.integrate(lim)
    for i, j in cartes(N, repeat=2):
        reps = dict(zip((a, b), (i, j)))
        assert ans.subs(reps) == p.subs(reps).integrate(lim)
    assert ans.subs(a, 4).subs(b, 1) == 0 + 2 * 3 + 1
    p = Piecewise((1, x > a), (2, x < b), (0, True))
    ans = p.integrate(lim)
    for i, j in cartes(N, repeat=2):
        reps = dict(zip((a, b), (i, j)))
        assert ans.subs(reps) == p.subs(reps).integrate(lim)