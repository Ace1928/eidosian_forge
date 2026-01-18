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
def test_pin_joint_arbitrary_axis():
    q, u = dynamicsymbols('q_J, u_J')
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=-A.x)
    assert (-A.x).angle_between(N.x) == 0
    assert -A.x.express(N) == N.x
    assert A.dcm(N) == Matrix([[-1, 0, 0], [0, -cos(q), -sin(q)], [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u * N.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)
    assert C.masscenter.pos_from(P.masscenter) == 0
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == 0
    assert C.masscenter.vel(N) == 0
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, child_interframe=A.y, child_point=A.x)
    assert A.y.angle_between(N.x) == 0
    assert A.y.express(N) == N.x
    assert A.dcm(N) == Matrix([[0, -cos(q), -sin(q)], [1, 0, 0], [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u * N.x
    assert A.ang_vel_in(N).express(A) == u * A.y
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)
    assert A.ang_vel_in(N).cross(A.y) == 0
    assert C.masscenter.vel(N) == u * A.z
    assert C.masscenter.pos_from(P.masscenter) == -A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == cos(q) * N.y + sin(q) * N.z
    assert C.masscenter.vel(N).angle_between(A.x) == pi / 2
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_interframe=N.y, parent_point=N.x)
    assert N.y.angle_between(A.x) == 0
    assert N.y.express(A) == A.x
    assert A.dcm(N) == Matrix([[0, 1, 0], [-cos(q), 0, sin(q)], [sin(q), 0, cos(q)]])
    assert A.ang_vel_in(N) == u * N.y
    assert A.ang_vel_in(N).express(A) == u * A.x
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)
    angle = A.ang_vel_in(N).angle_between(A.x)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N) == 0
    assert C.masscenter.pos_from(P.masscenter) == N.x
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x, child_interframe=A.x + A.y)
    assert expand_mul(N.x.angle_between(A.x + A.y)) == 0
    assert (A.x + A.y).express(N).simplify() == sqrt(2) * N.x
    assert _simplify_matrix(A.dcm(N)) == Matrix([[sqrt(2) / 2, -sqrt(2) * cos(q) / 2, -sqrt(2) * sin(q) / 2], [sqrt(2) / 2, sqrt(2) * cos(q) / 2, sqrt(2) * sin(q) / 2], [0, -sin(q), cos(q)]])
    assert A.ang_vel_in(N) == u * N.x
    assert A.ang_vel_in(N).express(A).simplify() == (u * A.x + u * A.y) / sqrt(2)
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N).simplify() == u * A.z / sqrt(2)
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == (1 - sqrt(2) / 2) * N.x + sqrt(2) * cos(q) / 2 * N.y + sqrt(2) * sin(q) / 2 * N.z
    assert C.masscenter.vel(N).express(N).simplify() == -sqrt(2) * u * sin(q) / 2 * N.y + sqrt(2) * u * cos(q) / 2 * N.z
    assert C.masscenter.vel(N).angle_between(A.x) == pi / 2
    N, A, P, C = _generate_body()
    PinJoint('J', P, C, parent_point=N.x, child_point=A.x, child_interframe=A.x + A.y - A.z)
    assert expand_mul(N.x.angle_between(A.x + A.y - A.z)) == 0
    assert (A.x + A.y - A.z).express(N).simplify() == sqrt(3) * N.x
    assert _simplify_matrix(A.dcm(N)) == Matrix([[sqrt(3) / 3, -sqrt(6) * sin(q + pi / 4) / 3, sqrt(6) * cos(q + pi / 4) / 3], [sqrt(3) / 3, sqrt(6) * cos(q + pi / 12) / 3, sqrt(6) * sin(q + pi / 12) / 3], [-sqrt(3) / 3, sqrt(6) * cos(q + 5 * pi / 12) / 3, sqrt(6) * sin(q + 5 * pi / 12) / 3]])
    assert A.ang_vel_in(N) == u * N.x
    assert A.ang_vel_in(N).express(A).simplify() == (u * A.x + u * A.y - u * A.z) / sqrt(3)
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y - A.z)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N).simplify() == (u * A.y + u * A.z) / sqrt(3)
    assert C.masscenter.pos_from(P.masscenter) == N.x - A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == (1 - sqrt(3) / 3) * N.x + sqrt(6) * sin(q + pi / 4) / 3 * N.y - sqrt(6) * cos(q + pi / 4) / 3 * N.z
    assert C.masscenter.vel(N).express(N).simplify() == sqrt(6) * u * cos(q + pi / 4) / 3 * N.y + sqrt(6) * u * sin(q + pi / 4) / 3 * N.z
    assert C.masscenter.vel(N).angle_between(A.x) == pi / 2
    N, A, P, C = _generate_body()
    m, n = symbols('m n')
    PinJoint('J', P, C, parent_point=m * N.x, child_point=n * A.x, child_interframe=A.x + A.y - A.z, parent_interframe=N.x - N.y + N.z)
    angle = (N.x - N.y + N.z).angle_between(A.x + A.y - A.z)
    assert expand_mul(angle) == 0
    assert (A.x - A.y + A.z).express(N).simplify() == (-4 * cos(q) / 3 - S(1) / 3) * N.x + (S(1) / 3 - 4 * sin(q + pi / 6) / 3) * N.y + (4 * cos(q + pi / 3) / 3 - S(1) / 3) * N.z
    assert _simplify_matrix(A.dcm(N)) == Matrix([[S(1) / 3 - 2 * cos(q) / 3, -2 * sin(q + pi / 6) / 3 - S(1) / 3, 2 * cos(q + pi / 3) / 3 + S(1) / 3], [2 * cos(q + pi / 3) / 3 + S(1) / 3, 2 * cos(q) / 3 - S(1) / 3, 2 * sin(q + pi / 6) / 3 + S(1) / 3], [-2 * sin(q + pi / 6) / 3 - S(1) / 3, 2 * cos(q + pi / 3) / 3 + S(1) / 3, 2 * cos(q) / 3 - S(1) / 3]])
    assert A.ang_vel_in(N) == (u * N.x - u * N.y + u * N.z) / sqrt(3)
    assert A.ang_vel_in(N).express(A).simplify() == (u * A.x + u * A.y - u * A.z) / sqrt(3)
    assert A.ang_vel_in(N).magnitude() == sqrt(u ** 2)
    angle = A.ang_vel_in(N).angle_between(A.x + A.y - A.z)
    assert angle.xreplace({u: 1}) == 0
    assert C.masscenter.vel(N).simplify() == sqrt(3) * n * u / 3 * A.y + sqrt(3) * n * u / 3 * A.z
    assert C.masscenter.pos_from(P.masscenter) == m * N.x - n * A.x
    assert C.masscenter.pos_from(P.masscenter).express(N).simplify() == (m + n * (2 * cos(q) - 1) / 3) * N.x + n * (2 * sin(q + pi / 6) + 1) / 3 * N.y - n * (2 * cos(q + pi / 3) + 1) / 3 * N.z
    assert C.masscenter.vel(N).express(N).simplify() == -2 * n * u * sin(q) / 3 * N.x + 2 * n * u * cos(q + pi / 6) / 3 * N.y + 2 * n * u * sin(q + pi / 3) / 3 * N.z
    assert C.masscenter.vel(N).dot(N.x - N.y + N.z).simplify() == 0