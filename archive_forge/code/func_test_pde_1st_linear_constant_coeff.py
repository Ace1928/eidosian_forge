from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pde_1st_linear_constant_coeff():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = -2 * u.diff(x) + 4 * u.diff(y) + 5 * u - exp(x + 3 * y)
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y), (F(4 * x + 2 * y) * exp(x / 2) + exp(x + 4 * y) / 15) * exp(-y))
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]
    eq = u.diff(x) / u + u.diff(y) / u + 1 - exp(x + y) / u
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y), F(x - y) * exp(-x / 2 - y / 2) + exp(x + y) / 3)
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]
    eq = 2 * u + -u.diff(x) + 3 * u.diff(y) + sin(x)
    sol = pdsolve(eq)
    assert sol == Eq(f(x, y), F(3 * x + y) * exp(x / 5 - 3 * y / 5) - 2 * sin(x) / 5 - cos(x) / 5)
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]
    eq = u + u.diff(x) + u.diff(y) + x * y
    sol = pdsolve(eq)
    assert sol.expand() == Eq(f(x, y), x + y + (x - y) ** 2 / 4 - (x + y) ** 2 / 4 + F(x - y) * exp(-x / 2 - y / 2) - 2).expand()
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')
    assert checkpdesol(eq, sol)[0]
    eq = u + u.diff(x) + u.diff(y) + log(x)
    assert classify_pde(eq) == ('1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral')