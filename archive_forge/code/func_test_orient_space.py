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
def test_orient_space():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_space_fixed(A, (0, 0, 0), '123')
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])