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
@staticmethod
def _call_derive_scalar_by_matrix(expr: Expr, v: MatrixCommon) -> Expr:
    return v.applyfunc(lambda x: expr.diff(x))