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
def test_piecewise_collapse():
    assert Piecewise((x, True)) == x
    a = x < 1
    assert Piecewise((x, a), (x + 1, a)) == Piecewise((x, a))
    assert Piecewise((x, a), (x + 1, a.reversed)) == Piecewise((x, a))
    b = x < 5

    def canonical(i):
        if isinstance(i, Piecewise):
            return Piecewise(*i.args)
        return i
    for args in [((1, a), (Piecewise((2, a), (3, b)), b)), ((1, a), (Piecewise((2, a), (3, b.reversed)), b)), ((1, a), (Piecewise((2, a), (3, b)), b), (4, True)), ((1, a), (Piecewise((2, a), (3, b), (4, True)), b)), ((1, a), (Piecewise((2, a), (3, b), (4, True)), b), (5, True))]:
        for i in (0, 2, 10):
            assert canonical(Piecewise(*args, evaluate=False).subs(x, i)) == canonical(Piecewise(*args).subs(x, i))
    r1, r2, r3, r4 = symbols('r1:5')
    a = x < r1
    b = x < r2
    c = x < r3
    d = x < r4
    assert Piecewise((1, a), (Piecewise((2, a), (3, b), (4, c)), b), (5, c)) == Piecewise((1, a), (3, b), (5, c))
    assert Piecewise((1, a), (Piecewise((2, a), (3, b), (4, c), (6, True)), c), (5, d)) == Piecewise((1, a), (Piecewise((3, b), (4, c)), c), (5, d))
    assert Piecewise((1, Or(a, d)), (Piecewise((2, d), (3, b), (4, c)), b), (5, c)) == Piecewise((1, Or(a, d)), (Piecewise((2, d), (3, b)), b), (5, c))
    assert Piecewise((1, c), (2, ~c), (3, S.true)) == Piecewise((1, c), (2, S.true))
    assert Piecewise((1, c), (2, And(~c, b)), (3, True)) == Piecewise((1, c), (2, b), (3, True))
    assert Piecewise((1, c), (2, Or(~c, b)), (3, True)).subs(dict(zip((r1, r2, r3, r4, x), (1, 2, 3, 4, 3.5)))) == 2
    assert Piecewise((1, c), (2, ~c)) == Piecewise((1, c), (2, True))