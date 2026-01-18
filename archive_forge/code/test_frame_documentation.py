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
This test ensures that the simplest form of that linear system solution
    is returned, thus the == for the expression comparison.