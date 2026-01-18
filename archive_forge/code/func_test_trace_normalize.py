from sympy.core import Lambda, S, symbols
from sympy.concrete import Sum
from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices import eye, Matrix, ShapeError, ImmutableMatrix
from sympy.matrices.expressions import (
from sympy.matrices.expressions.special import OneMatrix
from sympy.testing.pytest import raises
from sympy.abc import i
def test_trace_normalize():
    assert Trace(B * A) != Trace(A * B)
    assert Trace(B * A)._normalize() == Trace(A * B)
    assert Trace(B * A.T)._normalize() == Trace(A * B.T)