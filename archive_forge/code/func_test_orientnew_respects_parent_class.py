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
def test_orientnew_respects_parent_class():

    class MyReferenceFrame(ReferenceFrame):
        pass
    B = MyReferenceFrame('B')
    C = B.orientnew('C', 'Axis', [0, B.x])
    assert isinstance(C, MyReferenceFrame)