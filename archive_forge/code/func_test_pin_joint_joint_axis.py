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
def test_pin_joint_joint_axis():
    q, u = dynamicsymbols('q, u')
    N, A, P, C, Pint, Cint = _generate_body(True)
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint, joint_axis=P.y)
    assert pin.joint_axis == P.y
    assert N.dcm(A) == Matrix([[sin(q), 0, cos(q)], [0, -1, 0], [cos(q), 0, -sin(q)]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    pin = PinJoint('J', P, C, q, u, parent_interframe=Pint, child_interframe=Cint, joint_axis=Pint.y)
    assert pin.joint_axis == Pint.y
    assert N.dcm(A) == Matrix([[-sin(q), 0, cos(q)], [0, -1, 0], [cos(q), 0, sin(q)]])
    N, A, P, C = _generate_body()
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z, child_interframe=-C.z, joint_axis=N.z)
    assert pin.joint_axis == N.z
    assert N.dcm(A) == Matrix([[-cos(q), -sin(q), 0], [-sin(q), cos(q), 0], [0, 0, -1]])
    N, A, P, C = _generate_body()
    pin = PinJoint('J', P, C, q, u, parent_interframe=N.z, child_interframe=-C.z, joint_axis=N.x)
    assert pin.joint_axis == N.x
    assert N.dcm(A) == Matrix([[-1, 0, 0], [0, cos(q), sin(q)], [0, sin(q), -cos(q)]])
    N, A, P, C, Pint, Cint = _generate_body(True)
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=cos(q) * N.x + sin(q) * N.y))
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=C.x))
    raises(ValueError, lambda: PinJoint('J', P, C, joint_axis=P.x + C.y))
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint, joint_axis=Pint.x + C.y))
    raises(ValueError, lambda: PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint, joint_axis=P.x + Cint.y))
    N, A, P, C, Pint, Cint = _generate_body(True)
    PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint, joint_axis=Pint.x + P.y)
    raises(Exception, lambda: PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint, joint_axis=Vector(0)))
    raises(Exception, lambda: PinJoint('J', P, C, parent_interframe=Pint, child_interframe=Cint, joint_axis=P.y + Pint.y))