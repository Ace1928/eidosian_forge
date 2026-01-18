from sympy.core.relational import Eq
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.logic.boolalg import Boolean, And
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.common import ShapeError
from typing import Union
def validate_matmul_integer(*args: MatrixExpr) -> None:
    """Validate matrix shape for multiplication only for integer values"""
    for A, B in zip(args[:-1], args[1:]):
        i, j = (A.cols, B.rows)
        if isinstance(i, (int, Integer)) and isinstance(j, (int, Integer)) and (i != j):
            raise ShapeError('Matrices are not aligned', i, j)