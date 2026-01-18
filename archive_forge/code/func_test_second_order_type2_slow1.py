from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
def test_second_order_type2_slow1():
    x, y, z = symbols('x, y, z', cls=Function)
    t, l = symbols('t, l')
    eqs1 = [Eq(Derivative(x(t), (t, 2)), t * (2 * x(t) + y(t))), Eq(Derivative(y(t), (t, 2)), t * (-x(t) + 2 * y(t)))]
    sol1 = [Eq(x(t), I * C1 * airyai(t * (2 - I) ** (S(1) / 3)) + I * C2 * airybi(t * (2 - I) ** (S(1) / 3)) - I * C3 * airyai(t * (2 + I) ** (S(1) / 3)) - I * C4 * airybi(t * (2 + I) ** (S(1) / 3))), Eq(y(t), C1 * airyai(t * (2 - I) ** (S(1) / 3)) + C2 * airybi(t * (2 - I) ** (S(1) / 3)) + C3 * airyai(t * (2 + I) ** (S(1) / 3)) + C4 * airybi(t * (2 + I) ** (S(1) / 3)))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])