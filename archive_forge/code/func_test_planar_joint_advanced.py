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
def test_planar_joint_advanced():
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    l1, l2 = symbols('l1:3')
    N, A, P, C = _generate_body()
    J = PlanarJoint('J', P, C, q0, [q1, q2], u0, [u1, u2], parent_point=l1 * N.z, child_point=-l2 * C.z, parent_interframe=N.z + N.y / sqrt(3), child_interframe=A.z + A.y / sqrt(3))
    assert J.rotation_axis.express(N) == (N.z + N.y / sqrt(3)).normalize()
    assert J.rotation_axis.express(A) == (A.z + A.y / sqrt(3)).normalize()
    assert J.rotation_axis.angle_between(N.z) == pi / 6
    assert N.dcm(A).xreplace({q0: 0, q1: 0, q2: 0}) == eye(3)
    N_R_A = Matrix([[cos(q0), -sqrt(3) * sin(q0) / 2, sin(q0) / 2], [sqrt(3) * sin(q0) / 2, 3 * cos(q0) / 4 + 1 / 4, sqrt(3) * (1 - cos(q0)) / 4], [-sin(q0) / 2, sqrt(3) * (1 - cos(q0)) / 4, cos(q0) / 4 + 3 / 4]])
    assert _simplify_matrix(N.dcm(A) - N_R_A) == zeros(3)