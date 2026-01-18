from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_pendulum_angular_momentum():
    """Consider a pendulum of length OA = 2a, of mass m as a rigid body of
    center of mass G (OG = a) which turn around (O,z). The angle between the
    reference frame R and the rod is q.  The inertia of the body is I =
    (G,0,ma^2/3,ma^2/3). """
    m, a = symbols('m, a')
    q = dynamicsymbols('q')
    R = ReferenceFrame('R')
    R1 = R.orientnew('R1', 'Axis', [q, R.z])
    R1.set_ang_vel(R, q.diff() * R.z)
    I = inertia(R1, 0, m * a ** 2 / 3, m * a ** 2 / 3)
    O = Point('O')
    A = O.locatenew('A', 2 * a * R1.x)
    G = O.locatenew('G', a * R1.x)
    S = RigidBody('S', G, R1, m, (I, G))
    O.set_vel(R, 0)
    A.v2pt_theory(O, R, R1)
    G.v2pt_theory(O, R, R1)
    assert 4 * m * a ** 2 / 3 * q.diff() * R.z - S.angular_momentum(O, R).express(R) == 0