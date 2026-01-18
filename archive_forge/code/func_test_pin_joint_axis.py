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
def test_pin_joint_axis():
    q, u = dynamicsymbols('q u')
    N, A, P, C, Pint, Cint = _generate_body(True)
    J = PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint)
    assert J.joint_axis == Pint.x
    N_R_A = Matrix([[0, sin(q), cos(q)], [0, -cos(q), sin(q)], [1, 0, 0]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint, joint_axis=N.z)
    assert N.dcm(A) == N_R_A
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint, joint_axis=-Pint.z)
    assert N.dcm(A) == N_R_A
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=q * N.z))