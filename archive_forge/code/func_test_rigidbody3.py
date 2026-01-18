from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_rigidbody3():
    q1, q2, q3, q4 = dynamicsymbols('q1:5')
    p1, p2, p3 = symbols('p1:4')
    m = symbols('m')
    A = ReferenceFrame('A')
    B = A.orientnew('B', 'axis', [q1, A.x])
    O = Point('O')
    O.set_vel(A, q2 * A.x + q3 * A.y + q4 * A.z)
    P = O.locatenew('P', p1 * B.x + p2 * B.y + p3 * B.z)
    P.v2pt_theory(O, A, B)
    I = outer(B.x, B.x)
    rb1 = RigidBody('rb1', P, B, m, (I, P))
    rb2 = RigidBody('rb2', P, B, m, (I + inertia_of_point_mass(m, P.pos_from(O), B), O))
    assert rb1.central_inertia == rb2.central_inertia
    assert rb1.angular_momentum(O, A) == rb2.angular_momentum(O, A)