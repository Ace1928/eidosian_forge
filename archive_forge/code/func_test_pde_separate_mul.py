from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pde_separate_mul():
    x, y, z, t = symbols('x,y,z,t')
    c = Symbol('C', real=True)
    Phi = Function('Phi')
    F, R, T, X, Y, Z, u = map(Function, 'FRTXYZu')
    r, theta, z = symbols('r,theta,z')
    eq = Eq(D(F(x, y, z), x) + D(F(x, y, z), y) + D(F(x, y, z), z), 0)
    raises(ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), u(z, z)]))
    raises(ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(x), Y(y)]))
    raises(ValueError, lambda: pde_separate_mul(eq, F(x, y, z), [X(t), Y(x, y)]))
    assert pde_separate_mul(eq, F(x, y, z), [Y(y), u(x, z)]) == [D(Y(y), y) / Y(y), -D(u(x, z), x) / u(x, z) - D(u(x, z), z) / u(x, z)]
    assert pde_separate_mul(eq, F(x, y, z), [X(x), Y(y), Z(z)]) == [D(X(x), x) / X(x), -D(Z(z), z) / Z(z) - D(Y(y), y) / Y(y)]
    wave = Eq(D(u(x, t), t, t), c ** 2 * D(u(x, t), x, x))
    res = pde_separate_mul(wave, u(x, t), [X(x), T(t)])
    assert res == [D(X(x), x, x) / X(x), D(T(t), t, t) / (c ** 2 * T(t))]
    eq = Eq(1 / r * D(Phi(r, theta, z), r) + D(Phi(r, theta, z), r, 2) + 1 / r ** 2 * D(Phi(r, theta, z), theta, 2) + D(Phi(r, theta, z), z, 2), 0)
    res = pde_separate_mul(eq, Phi(r, theta, z), [Z(z), u(theta, r)])
    assert res == [D(Z(z), z, z) / Z(z), -D(u(theta, r), r, r) / u(theta, r) - D(u(theta, r), r) / (r * u(theta, r)) - D(u(theta, r), theta, theta) / (r ** 2 * u(theta, r))]
    eq = Eq(res[1], c)
    res = pde_separate_mul(eq, u(theta, r), [T(theta), R(r)])
    assert res == [D(T(theta), theta, theta) / T(theta), -r * D(R(r), r) / R(r) - r ** 2 * D(R(r), r, r) / R(r) - c * r ** 2]
    res = pde_separate_mul(eq, u(theta, r), [R(r), T(theta)])
    assert res == [r * D(R(r), r) / R(r) + r ** 2 * D(R(r), r, r) / R(r) + c * r ** 2, -D(T(theta), theta, theta) / T(theta)]