from collections import Counter
from sympy.core import Mul, sympify
from sympy.core.add import Add
from sympy.core.expr import ExprBuilder
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import log
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix
from sympy.strategies import (
from sympy.utilities.exceptions import sympy_deprecation_warning
class HadamardPower(MatrixExpr):
    """
    Elementwise power of matrix expressions

    Parameters
    ==========

    base : scalar or matrix

    exp : scalar or matrix

    Notes
    =====

    There are four definitions for the hadamard power which can be used.
    Let's consider `A, B` as `(m, n)` matrices, and `a, b` as scalars.

    Matrix raised to a scalar exponent:

    .. math::
        A^{\\circ b} = \\begin{bmatrix}
        A_{0, 0}^b   & A_{0, 1}^b   & \\cdots & A_{0, n-1}^b   \\\\
        A_{1, 0}^b   & A_{1, 1}^b   & \\cdots & A_{1, n-1}^b   \\\\
        \\vdots       & \\vdots       & \\ddots & \\vdots         \\\\
        A_{m-1, 0}^b & A_{m-1, 1}^b & \\cdots & A_{m-1, n-1}^b
        \\end{bmatrix}

    Scalar raised to a matrix exponent:

    .. math::
        a^{\\circ B} = \\begin{bmatrix}
        a^{B_{0, 0}}   & a^{B_{0, 1}}   & \\cdots & a^{B_{0, n-1}}   \\\\
        a^{B_{1, 0}}   & a^{B_{1, 1}}   & \\cdots & a^{B_{1, n-1}}   \\\\
        \\vdots         & \\vdots         & \\ddots & \\vdots           \\\\
        a^{B_{m-1, 0}} & a^{B_{m-1, 1}} & \\cdots & a^{B_{m-1, n-1}}
        \\end{bmatrix}

    Matrix raised to a matrix exponent:

    .. math::
        A^{\\circ B} = \\begin{bmatrix}
        A_{0, 0}^{B_{0, 0}}     & A_{0, 1}^{B_{0, 1}}     &
        \\cdots & A_{0, n-1}^{B_{0, n-1}}     \\\\
        A_{1, 0}^{B_{1, 0}}     & A_{1, 1}^{B_{1, 1}}     &
        \\cdots & A_{1, n-1}^{B_{1, n-1}}     \\\\
        \\vdots                  & \\vdots                  &
        \\ddots & \\vdots                      \\\\
        A_{m-1, 0}^{B_{m-1, 0}} & A_{m-1, 1}^{B_{m-1, 1}} &
        \\cdots & A_{m-1, n-1}^{B_{m-1, n-1}}
        \\end{bmatrix}

    Scalar raised to a scalar exponent:

    .. math::
        a^{\\circ b} = a^b
    """

    def __new__(cls, base, exp):
        base = sympify(base)
        exp = sympify(exp)
        if base.is_scalar and exp.is_scalar:
            return base ** exp
        if isinstance(base, MatrixExpr) and isinstance(exp, MatrixExpr):
            validate(base, exp)
        obj = super().__new__(cls, base, exp)
        return obj

    @property
    def base(self):
        return self._args[0]

    @property
    def exp(self):
        return self._args[1]

    @property
    def shape(self):
        if self.base.is_Matrix:
            return self.base.shape
        return self.exp.shape

    def _entry(self, i, j, **kwargs):
        base = self.base
        exp = self.exp
        if base.is_Matrix:
            a = base._entry(i, j, **kwargs)
        elif base.is_scalar:
            a = base
        else:
            raise ValueError('The base {} must be a scalar or a matrix.'.format(base))
        if exp.is_Matrix:
            b = exp._entry(i, j, **kwargs)
        elif exp.is_scalar:
            b = exp
        else:
            raise ValueError('The exponent {} must be a scalar or a matrix.'.format(exp))
        return a ** b

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardPower(transpose(self.base), self.exp)

    def _eval_derivative(self, x):
        dexp = self.exp.diff(x)
        logbase = self.base.applyfunc(log)
        dlbase = logbase.diff(x)
        return hadamard_product(dexp * logbase + self.exp * dlbase, self)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.matrices.expressions.matexpr import _make_matrix
        lr = self.base._eval_derivative_matrix_lines(x)
        for i in lr:
            diagonal = [(1, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.base.shape[j] != 1]
            l1 = i._lines[i._first_line_index]
            l2 = i._lines[i._second_line_index]
            subexpr = ExprBuilder(ArrayDiagonal, [ExprBuilder(ArrayTensorProduct, [ExprBuilder(_make_matrix, [l1]), self.exp * hadamard_power(self.base, self.exp - 1), ExprBuilder(_make_matrix, [l2])]), *diagonal], validator=ArrayDiagonal._validate)
            i._first_pointer_parent = subexpr.args[0].args[0].args
            i._first_pointer_index = 0
            i._first_line_index = 0
            i._second_pointer_parent = subexpr.args[0].args[2].args
            i._second_pointer_index = 0
            i._second_line_index = 0
            i._lines = [subexpr]
        return lr