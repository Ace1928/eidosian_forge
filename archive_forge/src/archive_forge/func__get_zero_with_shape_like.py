from __future__ import annotations
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.numbers import Integer
from sympy.matrices.common import MatrixCommon
from .ndim_array import NDimArray
from .arrayop import derive_by_array
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.matrices.expressions.matexpr import _matrix_derivative
@classmethod
def _get_zero_with_shape_like(cls, expr):
    if isinstance(expr, (MatrixCommon, NDimArray)):
        return expr.zeros(*expr.shape)
    elif isinstance(expr, MatrixExpr):
        return ZeroMatrix(*expr.shape)
    else:
        raise RuntimeError('Unable to determine shape of array-derivative.')