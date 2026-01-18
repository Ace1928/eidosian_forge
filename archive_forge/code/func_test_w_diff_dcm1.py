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
def test_w_diff_dcm1():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    c11, c12, c13 = dynamicsymbols('C11 C12 C13')
    c21, c22, c23 = dynamicsymbols('C21 C22 C23')
    c31, c32, c33 = dynamicsymbols('C31 C32 C33')
    c11d, c12d, c13d = dynamicsymbols('C11 C12 C13', level=1)
    c21d, c22d, c23d = dynamicsymbols('C21 C22 C23', level=1)
    c31d, c32d, c33d = dynamicsymbols('C31 C32 C33', level=1)
    DCM = Matrix([[c11, c12, c13], [c21, c22, c23], [c31, c32, c33]])
    B.orient(A, 'DCM', DCM)
    b1a = B.x.express(A)
    b2a = B.y.express(A)
    b3a = B.z.express(A)
    B.set_ang_vel(A, B.x * dot(b3a.dt(A), B.y) + B.y * dot(b1a.dt(A), B.z) + B.z * dot(b2a.dt(A), B.x))
    expr = (c12 * c13d + c22 * c23d + c32 * c33d) * B.x + (c13 * c11d + c23 * c21d + c33 * c31d) * B.y + (c11 * c12d + c21 * c22d + c31 * c32d) * B.z
    assert B.ang_vel_in(A) - expr == 0