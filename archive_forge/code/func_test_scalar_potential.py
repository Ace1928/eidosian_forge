from sympy.core.function import Derivative
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.simplify import simplify
from sympy.core.symbol import symbols
from sympy.core import S
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.vector.vector import Dot
from sympy.vector.operators import curl, divergence, gradient, Gradient, Divergence, Cross
from sympy.vector.deloperator import Del
from sympy.vector.functions import (is_conservative, is_solenoidal,
from sympy.testing.pytest import raises
def test_scalar_potential():
    assert scalar_potential(Vector.zero, C) == 0
    assert scalar_potential(i, C) == x
    assert scalar_potential(j, C) == y
    assert scalar_potential(k, C) == z
    assert scalar_potential(y * z * i + x * z * j + x * y * k, C) == x * y * z
    assert scalar_potential(grad_field, C) == scalar_field
    assert scalar_potential(z * P.i + P.x * k, C) == x * z * cos(q) + y * z * sin(q)
    assert scalar_potential(z * P.i + P.x * k, P) == P.x * P.z
    raises(ValueError, lambda: scalar_potential(x * j, C))