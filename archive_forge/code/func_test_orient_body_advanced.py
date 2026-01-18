from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
def test_orient_body_advanced():
    q1, q2, q3 = dynamicsymbols('q1:4')
    c1, c2, c3 = symbols('c1:4')
    u1, u2, u3 = dynamicsymbols('q1:4', 1)
    A, B = (ReferenceFrame('A'), ReferenceFrame('B'))
    B.orient_body_fixed(A, (q1, q2, q3), 'zxy')
    assert A.dcm(B) == Matrix([[-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)], [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([[-sin(q3) * cos(q2) * u1 + cos(q3) * u2], [sin(q2) * u1 + u3], [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])
    A, B = (ReferenceFrame('A'), ReferenceFrame('B'))
    B.orient_body_fixed(A, (q1, c2, q3), 131)
    assert A.dcm(B) == Matrix([[cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)], [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3), -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)], [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1), -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([[cos(c2) * u1 + u3], [-sin(c2) * cos(q3) * u1], [sin(c2) * sin(q3) * u1]])
    A, B = (ReferenceFrame('A'), ReferenceFrame('B'))
    B.orient_body_fixed(A, (c1, c2, c3), 123)
    assert B.ang_vel_in(A) == Vector(0)