from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_point_v2pt_theorys():
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    N = ReferenceFrame('N')
    B = N.orientnew('B', 'Axis', [q, N.z])
    O = Point('O')
    P = O.locatenew('P', 0)
    O.set_vel(N, 0)
    assert P.v2pt_theory(O, N, B) == 0
    P = O.locatenew('P', B.x)
    assert P.v2pt_theory(O, N, B) == qd * B.z ^ B.x
    O.set_vel(N, N.x)
    assert P.v2pt_theory(O, N, B) == N.x + qd * B.y