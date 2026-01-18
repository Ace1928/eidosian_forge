from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_rigidbody():
    m, m2, v1, v2, v3, omega = symbols('m m2 v1 v2 v3 omega')
    A = ReferenceFrame('A')
    A2 = ReferenceFrame('A2')
    P = Point('P')
    P2 = Point('P2')
    I = Dyadic(0)
    I2 = Dyadic(0)
    B = RigidBody('B', P, A, m, (I, P))
    assert B.mass == m
    assert B.frame == A
    assert B.masscenter == P
    assert B.inertia == (I, B.masscenter)
    B.mass = m2
    B.frame = A2
    B.masscenter = P2
    B.inertia = (I2, B.masscenter)
    raises(TypeError, lambda: RigidBody(P, P, A, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, P, m, (I, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (P, P)))
    raises(TypeError, lambda: RigidBody('B', P, A, m, (I, I)))
    assert B.__str__() == 'B'
    assert B.mass == m2
    assert B.frame == A2
    assert B.masscenter == P2
    assert B.inertia == (I2, B.masscenter)
    assert B.masscenter == P2
    assert B.inertia == (I2, B.masscenter)
    N = ReferenceFrame('N')
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert B.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)