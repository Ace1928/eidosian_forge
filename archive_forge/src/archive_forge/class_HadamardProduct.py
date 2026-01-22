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
class HadamardProduct(MatrixExpr):
    """
    Elementwise product of matrix expressions

    Examples
    ========

    Hadamard product for matrix symbols:

    >>> from sympy import hadamard_product, HadamardProduct, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> isinstance(hadamard_product(A, B), HadamardProduct)
    True

    Notes
    =====

    This is a symbolic object that simply stores its argument without
    evaluating it. To actually compute the product, use the function
    ``hadamard_product()`` or ``HadamardProduct.doit``
    """
    is_HadamardProduct = True

    def __new__(cls, *args, evaluate=False, check=None):
        args = list(map(sympify, args))
        if len(args) == 0:
            raise ValueError('HadamardProduct needs at least one argument')
        if not all((isinstance(arg, MatrixExpr) for arg in args)):
            raise TypeError('Mix of Matrix and Scalar symbols')
        if check is not None:
            sympy_deprecation_warning('Passing check to HadamardProduct is deprecated and the check argument will be removed in a future version.', deprecated_since_version='1.11', active_deprecations_target='remove-check-argument-from-matrix-operations')
        if check is not False:
            validate(*args)
        obj = super().__new__(cls, *args)
        if evaluate:
            obj = obj.doit(deep=False)
        return obj

    @property
    def shape(self):
        return self.args[0].shape

    def _entry(self, i, j, **kwargs):
        return Mul(*[arg._entry(i, j, **kwargs) for arg in self.args])

    def _eval_transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return HadamardProduct(*list(map(transpose, self.args)))

    def doit(self, **hints):
        expr = self.func(*(i.doit(**hints) for i in self.args))
        from sympy.matrices.matrices import MatrixBase
        from sympy.matrices.immutable import ImmutableMatrix
        explicit = [i for i in expr.args if isinstance(i, MatrixBase)]
        if explicit:
            remainder = [i for i in expr.args if i not in explicit]
            expl_mat = ImmutableMatrix([Mul.fromiter(i) for i in zip(*explicit)]).reshape(*self.shape)
            expr = HadamardProduct(*[expl_mat] + remainder)
        return canonicalize(expr)

    def _eval_derivative(self, x):
        terms = []
        args = list(self.args)
        for i in range(len(args)):
            factors = args[:i] + [args[i].diff(x)] + args[i + 1:]
            terms.append(hadamard_product(*factors))
        return Add.fromiter(terms)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        from sympy.matrices.expressions.matexpr import _make_matrix
        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind + 1:]
            d = self.args[ind]._eval_derivative_matrix_lines(x)
            hadam = hadamard_product(*right_args + left_args)
            diagonal = [(0, 2), (3, 4)]
            diagonal = [e for j, e in enumerate(diagonal) if self.shape[j] != 1]
            for i in d:
                l1 = i._lines[i._first_line_index]
                l2 = i._lines[i._second_line_index]
                subexpr = ExprBuilder(ArrayDiagonal, [ExprBuilder(ArrayTensorProduct, [ExprBuilder(_make_matrix, [l1]), hadam, ExprBuilder(_make_matrix, [l2])]), *diagonal])
                i._first_pointer_parent = subexpr.args[0].args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args[2].args
                i._second_pointer_index = 0
                i._lines = [subexpr]
                lines.append(i)
        return lines