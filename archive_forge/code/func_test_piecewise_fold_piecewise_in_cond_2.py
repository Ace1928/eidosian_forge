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
def test_piecewise_fold_piecewise_in_cond_2():
    p1 = Piecewise((cos(x), x < 0), (0, True))
    p2 = Piecewise((0, Eq(p1, 0)), (1 / p1, True))
    p3 = Piecewise((0, (x >= 0) | Eq(cos(x), 0)), (1 / cos(x), x < 0), (zoo, True))
    assert piecewise_fold(p2) == p3