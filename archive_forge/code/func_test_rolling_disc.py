from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test_rolling_disc():
    q1, q2, q3 = dynamicsymbols('q1 q2 q3')
    q1d, q2d, q3d = dynamicsymbols('q1 q2 q3', 1)
    r, m, g = symbols('r m g')
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    C = Point('C')
    C.set_vel(N, 0)
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)
    I = inertia(L, m / 4 * r ** 2, m / 2 * r ** 2, m / 4 * r ** 2)
    BodyD = RigidBody('BodyD', Dmc, R, m, (I, Dmc))
    BodyD.potential_energy = -m * g * r * cos(q2)
    Lag = Lagrangian(N, BodyD)
    q = [q1, q2, q3]
    q1 = Function('q1')
    q2 = Function('q2')
    q3 = Function('q3')
    l = LagrangesMethod(Lag, q)
    l.form_lagranges_equations()
    RHS = l.rhs()
    RHS.simplify()
    t = symbols('t')
    assert l.mass_matrix[3:6] == [0, 5 * m * r ** 2 / 4, 0]
    assert RHS[4].simplify() == (-8 * g * sin(q2(t)) + r * (5 * sin(2 * q2(t)) * Derivative(q1(t), t) + 12 * cos(q2(t)) * Derivative(q3(t), t)) * Derivative(q1(t), t)) / (10 * r)
    assert RHS[5] == (-5 * cos(q2(t)) * Derivative(q1(t), t) + 6 * tan(q2(t)) * Derivative(q3(t), t) + 4 * Derivative(q1(t), t) / cos(q2(t))) * Derivative(q2(t), t)