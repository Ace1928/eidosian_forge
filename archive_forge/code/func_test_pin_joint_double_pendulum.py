from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core.backend import Matrix, _simplify_matrix, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (dynamicsymbols, Body, JointsMethod,
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_pin_joint_double_pendulum():
    q1, q2 = dynamicsymbols('q1 q2')
    u1, u2 = dynamicsymbols('u1 u2')
    m, l = symbols('m l')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = Body('C', frame=N)
    PartP = Body('P', frame=A, mass=m)
    PartR = Body('R', frame=B, mass=m)
    J1 = PinJoint('J1', C, PartP, speeds=u1, coordinates=q1, child_point=-l * A.x, joint_axis=C.frame.z)
    J2 = PinJoint('J2', PartP, PartR, speeds=u2, coordinates=q2, child_point=-l * B.x, joint_axis=PartP.frame.z)
    assert N.dcm(A) == Matrix([[cos(q1), -sin(q1), 0], [sin(q1), cos(q1), 0], [0, 0, 1]])
    assert A.dcm(B) == Matrix([[cos(q2), -sin(q2), 0], [sin(q2), cos(q2), 0], [0, 0, 1]])
    assert _simplify_matrix(N.dcm(B)) == Matrix([[cos(q1 + q2), -sin(q1 + q2), 0], [sin(q1 + q2), cos(q1 + q2), 0], [0, 0, 1]])
    assert A.ang_vel_in(N) == u1 * N.z
    assert B.ang_vel_in(A) == u2 * A.z
    assert B.ang_vel_in(N) == u1 * N.z + u2 * A.z
    assert J1.kdes == Matrix([u1 - q1.diff(t)])
    assert J2.kdes == Matrix([u2 - q2.diff(t)])
    assert PartP.masscenter.vel(N) == l * u1 * A.y
    assert PartR.masscenter.vel(A) == l * u2 * B.y
    assert PartR.masscenter.vel(N) == l * u1 * A.y + l * (u1 + u2) * B.y