from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_solvefun():
    f, F, G, H = map(Function, ['f', 'F', 'G', 'H'])
    eq1 = f(x, y) + f(x, y).diff(x) + f(x, y).diff(y)
    assert pdsolve(eq1) == Eq(f(x, y), F(x - y) * exp(-x / 2 - y / 2))
    assert pdsolve(eq1, solvefun=G) == Eq(f(x, y), G(x - y) * exp(-x / 2 - y / 2))
    assert pdsolve(eq1, solvefun=H) == Eq(f(x, y), H(x - y) * exp(-x / 2 - y / 2))