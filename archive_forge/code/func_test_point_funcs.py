from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_point_funcs():
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    B.set_ang_vel(N, 5 * B.y)
    O = Point('O')
    P = O.locatenew('P', q * B.x)
    assert P.pos_from(O) == q * B.x
    P.set_vel(B, qd * B.x + q2d * B.y)
    assert P.vel(B) == qd * B.x + q2d * B.y
    O.set_vel(N, 0)
    assert O.vel(N) == 0
    assert P.a1pt_theory(O, N, B) == (-25 * q + qdd) * B.x + q2dd * B.y + -10 * qd * B.z
    B = N.orientnew('B', 'Axis', [q, N.z])
    O = Point('O')
    P = O.locatenew('P', 10 * B.x)
    O.set_vel(N, 5 * N.x)
    assert O.vel(N) == 5 * N.x
    assert P.a2pt_theory(O, N, B) == -10 * qd ** 2 * B.x + 10 * qdd * B.y
    B.set_ang_vel(N, 5 * B.y)
    O = Point('O')
    P = O.locatenew('P', q * B.x)
    P.set_vel(B, qd * B.x + q2d * B.y)
    O.set_vel(N, 0)
    assert P.v1pt_theory(O, N, B) == qd * B.x + q2d * B.y - 5 * q * B.z