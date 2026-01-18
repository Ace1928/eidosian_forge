from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_vector_xreplace():
    x, y, z = symbols('x y z')
    v = x ** 2 * A.x + x * y * A.y + x * y * z * A.z
    assert v.xreplace({x: cos(x)}) == cos(x) ** 2 * A.x + y * cos(x) * A.y + y * z * cos(x) * A.z
    assert v.xreplace({x * y: pi}) == x ** 2 * A.x + pi * A.y + x * y * z * A.z
    assert v.xreplace({x * y * z: 1}) == x ** 2 * A.x + x * y * A.y + A.z
    assert v.xreplace({x: 1, z: 0}) == A.x + y * A.y
    raises(TypeError, lambda: v.xreplace())
    raises(TypeError, lambda: v.xreplace([x, y]))