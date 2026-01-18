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
def test_spherical_joint_coords():
    q0s, q1s, q2s, u0s, u1s, u2s = dynamicsymbols('q0:3_S, u0:3_S')
    q0, q1, q2, q3, u0, u1, u2, u4 = dynamicsymbols('q0:4, u0:4')
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, [q0, q1, q2], [u0, u1, u2])
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    N, A, P, C = _generate_body()
    S = SphericalJoint('S', P, C, Matrix([q0, q1, q2]), Matrix([u0, u1, u2]))
    assert S.coordinates == Matrix([q0, q1, q2])
    assert S.speeds == Matrix([u0, u1, u2])
    N, A, P, C = _generate_body()
    raises(ValueError, lambda: SphericalJoint('S', P, C, Matrix([q0, q1]), Matrix([u0])))
    raises(ValueError, lambda: SphericalJoint('S', P, C, Matrix([q0, q1, q2, q3]), Matrix([u0, u1, u2])))
    raises(ValueError, lambda: SphericalJoint('S', P, C, Matrix([q0, q1, q2]), Matrix([u0, u1, u2, u4])))