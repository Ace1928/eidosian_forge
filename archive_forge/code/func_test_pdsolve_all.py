from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pdsolve_all():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = u + u.diff(x) + u.diff(y) + x ** 2 * y
    sol = pdsolve(eq, hint='all')
    keys = ['1st_linear_constant_coeff', '1st_linear_constant_coeff_Integral', 'default', 'order']
    assert sorted(sol.keys()) == keys
    assert sol['order'] == 1
    assert sol['default'] == '1st_linear_constant_coeff'
    assert sol['1st_linear_constant_coeff'].expand() == Eq(f(x, y), -x ** 2 * y + x ** 2 + 2 * x * y - 4 * x - 2 * y + F(x - y) * exp(-x / 2 - y / 2) + 6).expand()