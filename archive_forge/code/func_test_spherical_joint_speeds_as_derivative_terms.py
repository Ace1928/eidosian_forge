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
def test_spherical_joint_speeds_as_derivative_terms():
    q0, q1, q2 = dynamicsymbols('q0:3')
    u0, u1, u2 = dynamicsymbols('q0:3', 1)
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, coordinates=[q0, q1, q2], speeds=[u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    assert S.kdes == Matrix([0, 0, 0])
    assert P.ang_vel_in(C) == (-u0 * cos(q1) * cos(q2) - u1 * sin(q2)) * A.x + (u0 * sin(q2) * cos(q1) - u1 * cos(q2)) * A.y + (-u0 * sin(q1) - u2) * A.z