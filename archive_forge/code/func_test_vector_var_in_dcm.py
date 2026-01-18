from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_vector_var_in_dcm():
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')
    v = u1 * u2 * A.x + u3 * N.y + u4 ** 2 * N.z
    assert v.diff(u1, N, var_in_dcm=False) == u2 * A.x
    assert v.diff(u1, A, var_in_dcm=False) == u2 * A.x
    assert v.diff(u3, N, var_in_dcm=False) == N.y
    assert v.diff(u3, A, var_in_dcm=False) == N.y
    assert v.diff(u3, B, var_in_dcm=False) == N.y
    assert v.diff(u4, N, var_in_dcm=False) == 2 * u4 * N.z
    raises(ValueError, lambda: v.diff(u1, N))