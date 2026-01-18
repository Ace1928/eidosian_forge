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
def test_spherical_joint_orient_body():
    q0, q1, q2, u0, u1, u2 = dynamicsymbols('q0:3, u0:3')
    N_R_A = Matrix([[-sin(q1), -sin(q2) * cos(q1), cos(q1) * cos(q2)], [-sin(q0) * cos(q1), sin(q0) * sin(q1) * sin(q2) - cos(q0) * cos(q2), -sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0)], [cos(q0) * cos(q1), -sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0), -sin(q0) * sin(q2) + sin(q1) * cos(q0) * cos(q2)]])
    N_w_A = Matrix([[-u0 * sin(q1) - u2], [-u0 * sin(q2) * cos(q1) + u1 * cos(q2)], [u0 * cos(q1) * cos(q2) + u1 * sin(q2)]])
    N_v_Co = Matrix([[-sqrt(2) * (u0 * cos(q2 + pi / 4) * cos(q1) + u1 * sin(q2 + pi / 4))], [-u0 * sin(q1) - u2], [-u0 * sin(q1) - u2]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2], parent_point=N.x + N.y, child_point=-A.y + A.z, parent_interframe=Pint, child_interframe=Cint, rot_type='body', rot_order=123)
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert _simplify_matrix(N.dcm(A) - N_R_A) == zeros(3)
    assert A.ang_vel_in(N).to_matrix(A) == N_w_A
    assert C.masscenter.vel(N).to_matrix(A) == N_v_Co
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2], parent_point=N.x + N.y, child_point=-A.y + A.z, parent_interframe=Pint, child_interframe=Cint, rot_type='BODY', amounts=(q1, q0, q2), rot_order=123)
    switch_order = lambda expr: expr.xreplace({q0: q1, q1: q0, q2: q2, u0: u1, u1: u0, u2: u2})
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 123
    assert _simplify_matrix(N.dcm(A) - switch_order(N_R_A)) == zeros(3)
    assert A.ang_vel_in(N).to_matrix(A) == switch_order(N_w_A)
    assert C.masscenter.vel(N).to_matrix(A) == switch_order(N_v_Co)
    N, A, P, C, Pint, Cint = _generate_body(True)
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2], parent_point=N.x + N.y, child_point=-A.y + A.z, parent_interframe=Pint, child_interframe=Cint, rot_type='BodY', rot_order='yxz')
    assert S._rot_type.upper() == 'BODY'
    assert S._rot_order == 'yxz'
    assert _simplify_matrix(N.dcm(A) - Matrix([[-sin(q0) * cos(q1), sin(q0) * sin(q1) * cos(q2) - sin(q2) * cos(q0), sin(q0) * sin(q1) * sin(q2) + cos(q0) * cos(q2)], [-sin(q1), -cos(q1) * cos(q2), -sin(q2) * cos(q1)], [cos(q0) * cos(q1), -sin(q0) * sin(q2) - sin(q1) * cos(q0) * cos(q2), sin(q0) * cos(q2) - sin(q1) * sin(q2) * cos(q0)]])) == zeros(3)
    assert A.ang_vel_in(N).to_matrix(A) == Matrix([[u0 * sin(q1) - u2], [u0 * cos(q1) * cos(q2) - u1 * sin(q2)], [u0 * sin(q2) * cos(q1) + u1 * cos(q2)]])
    assert C.masscenter.vel(N).to_matrix(A) == Matrix([[-sqrt(2) * (u0 * sin(q2 + pi / 4) * cos(q1) + u1 * cos(q2 + pi / 4))], [u0 * sin(q1) - u2], [u0 * sin(q1) - u2]])