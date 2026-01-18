from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
from sympy.testing.pytest import raises
def test_time_derivative():
    A = ReferenceFrame('A')
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    B = A.orientnew('B', 'Axis', [q, A.z])
    d = A.x | A.x
    assert time_derivative(d, B) == -qd * (A.y | A.x) + -qd * (A.x | A.y)
    d1 = A.x | B.y
    assert time_derivative(d1, A) == -qd * (A.x | B.x)
    assert time_derivative(d1, B) == -qd * (A.y | B.y)
    d2 = A.x | B.x
    assert time_derivative(d2, A) == qd * (A.x | B.y)
    assert time_derivative(d2, B) == -qd * (A.y | B.x)
    d3 = A.x | B.z
    assert time_derivative(d3, A) == 0
    assert time_derivative(d3, B) == -qd * (A.y | B.z)
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)
    C = B.orientnew('C', 'Axis', [q4, B.x])
    v1 = q1 * A.z
    v2 = q2 * A.x + q3 * B.y
    v3 = q1 * A.x + q2 * A.y + q3 * A.z
    assert time_derivative(B.x, C) == 0
    assert time_derivative(B.y, C) == -q4d * B.z
    assert time_derivative(B.z, C) == q4d * B.y
    assert time_derivative(v1, B) == q1d * A.z
    assert time_derivative(v1, C) == -q1 * sin(q) * q4d * A.x + q1 * cos(q) * q4d * A.y + q1d * A.z
    assert time_derivative(v2, A) == q2d * A.x - q3 * qd * B.x + q3d * B.y
    assert time_derivative(v2, C) == q2d * A.x - q2 * qd * A.y + q2 * sin(q) * q4d * A.z + q3d * B.y - q3 * q4d * B.z
    assert time_derivative(v3, B) == (q2 * qd + q1d) * A.x + (-q1 * qd + q2d) * A.y + q3d * A.z
    assert time_derivative(d, C) == -qd * (A.y | A.x) + sin(q) * q4d * (A.z | A.x) - qd * (A.x | A.y) + sin(q) * q4d * (A.x | A.z)
    raises(ValueError, lambda: time_derivative(B.x, C, order=0.5))
    raises(ValueError, lambda: time_derivative(B.x, C, order=-1))