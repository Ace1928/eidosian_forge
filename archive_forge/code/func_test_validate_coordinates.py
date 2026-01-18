from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_validate_coordinates():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1:4 u1:4')
    s1, s2, s3 = symbols('s1:4')
    _validate_coordinates([q1, q2, q3], [u1, u2, u3])
    _validate_coordinates([q1, q2])
    _validate_coordinates([q1, q2], [u1])
    _validate_coordinates(speeds=[u1, u2])
    _validate_coordinates([q1, q2, q2], [u1, u2, u3], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q2], [u1, u2, u3]))
    _validate_coordinates([q1, q2, q3], [u1, u2, u2], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [u1, u2, u2], check_duplicates=True))
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [q1, u2, u3], check_duplicates=True))
    _validate_coordinates([q1 + q2, q3], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([q1 + q2, q3]))
    _validate_coordinates([s1, q1, q2], [0, u1, u2], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([s1, q1, q2], [0, u1, u2], is_dynamicsymbols=True))
    _validate_coordinates([s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=True))
    t = dynamicsymbols._t
    a = symbols('a')
    f1, f2 = symbols('f1:3', cls=Function)
    _validate_coordinates([f1(a), f2(a)], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([f1(a), f2(a)]))
    raises(ValueError, lambda: _validate_coordinates(speeds=[f1(a), f2(a)]))
    dynamicsymbols._t = a
    _validate_coordinates([f1(a), f2(a)])
    raises(ValueError, lambda: _validate_coordinates([f1(t), f2(t)]))
    dynamicsymbols._t = t