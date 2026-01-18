from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises

We have a few different kind of Matrices
Matrix, ImmutableMatrix, MatrixExpr

Here we test the extent to which they cooperate
