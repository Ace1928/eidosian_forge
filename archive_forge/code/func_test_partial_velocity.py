from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
from sympy.testing.pytest import raises
def test_partial_velocity():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    u4, u5 = dynamicsymbols('u4, u5')
    r = symbols('r')
    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)
    C = Point('C')
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)
    vel_list = [Dmc.vel(N), C.vel(N), R.ang_vel_in(N)]
    u_list = [u1, u2, u3, u4, u5]
    assert partial_velocity(vel_list, u_list, N) == [[-r * L.y, r * L.x, 0, L.x, cos(q2) * L.y - sin(q2) * L.z], [0, 0, 0, L.x, cos(q2) * L.y - sin(q2) * L.z], [L.x, L.y, L.z, 0, 0]]
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    v = u4 * A.x + u5 * B.y
    assert partial_velocity((v,), (u4, u5), A) == [[A.x, B.y]]
    raises(TypeError, lambda: partial_velocity(Dmc.vel(N), u_list, N))
    raises(TypeError, lambda: partial_velocity(vel_list, u1, N))