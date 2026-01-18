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
def test_weld_joint():
    _, _, P, C = _generate_body()
    W = WeldJoint('W', P, C)
    assert W.name == 'W'
    assert W.parent == P
    assert W.child == C
    assert W.coordinates == Matrix()
    assert W.speeds == Matrix()
    assert W.kdes == Matrix(1, 0, []).T
    assert P.dcm(C) == eye(3)
    assert W.child_point.pos_from(C.masscenter) == Vector(0)
    assert W.parent_point.pos_from(P.masscenter) == Vector(0)
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == Vector(0)
    assert C.masscenter.vel(P.frame) == Vector(0)
    assert P.ang_vel_in(C) == 0
    assert C.ang_vel_in(P) == 0
    assert W.__str__() == 'WeldJoint: W  parent: P  child: C'
    N, A, P, C = _generate_body()
    l, m = symbols('l m')
    Pint = ReferenceFrame('P_int')
    Pint.orient_axis(P.frame, P.y, pi / 2)
    W = WeldJoint('W', P, C, parent_point=l * P.frame.x, child_point=m * C.frame.y, parent_interframe=Pint)
    assert W.child_point.pos_from(C.masscenter) == m * C.frame.y
    assert W.parent_point.pos_from(P.masscenter) == l * P.frame.x
    assert W.parent_point.pos_from(W.child_point) == Vector(0)
    assert P.masscenter.pos_from(C.masscenter) == -l * N.x + m * A.y
    assert C.masscenter.vel(P.frame) == Vector(0)
    assert P.masscenter.vel(Pint) == Vector(0)
    assert C.ang_vel_in(P) == 0
    assert P.ang_vel_in(C) == 0
    assert P.x == A.z
    JointsMethod(P, W)