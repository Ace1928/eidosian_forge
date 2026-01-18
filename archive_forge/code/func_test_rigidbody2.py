from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_rigidbody2():
    M, v, r, omega, g, h = dynamicsymbols('M v r omega g h')
    N = ReferenceFrame('N')
    b = ReferenceFrame('b')
    b.set_ang_vel(N, omega * b.x)
    P = Point('P')
    I = outer(b.x, b.x)
    Inertia_tuple = (I, P)
    B = RigidBody('B', P, b, M, Inertia_tuple)
    P.set_vel(N, v * b.x)
    assert B.angular_momentum(P, N) == omega * b.x
    O = Point('O')
    O.set_vel(N, v * b.x)
    P.set_pos(O, r * b.y)
    assert B.angular_momentum(O, N) == omega * b.x - M * v * r * b.z
    B.potential_energy = M * g * h
    assert B.potential_energy == M * g * h
    assert expand(2 * B.kinetic_energy(N)) == omega ** 2 + M * v ** 2