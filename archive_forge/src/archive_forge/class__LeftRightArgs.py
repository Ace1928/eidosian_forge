from __future__ import annotations
from functools import wraps
from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.matrices import MatrixKind, MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
class _LeftRightArgs:
    """
    Helper class to compute matrix derivatives.

    The logic: when an expression is derived by a matrix `X_{mn}`, two lines of
    matrix multiplications are created: the one contracted to `m` (first line),
    and the one contracted to `n` (second line).

    Transposition flips the side by which new matrices are connected to the
    lines.

    The trace connects the end of the two lines.
    """

    def __init__(self, lines, higher=S.One):
        self._lines = list(lines)
        self._first_pointer_parent = self._lines
        self._first_pointer_index = 0
        self._first_line_index = 0
        self._second_pointer_parent = self._lines
        self._second_pointer_index = 1
        self._second_line_index = 1
        self.higher = higher

    @property
    def first_pointer(self):
        return self._first_pointer_parent[self._first_pointer_index]

    @first_pointer.setter
    def first_pointer(self, value):
        self._first_pointer_parent[self._first_pointer_index] = value

    @property
    def second_pointer(self):
        return self._second_pointer_parent[self._second_pointer_index]

    @second_pointer.setter
    def second_pointer(self, value):
        self._second_pointer_parent[self._second_pointer_index] = value

    def __repr__(self):
        built = [self._build(i) for i in self._lines]
        return '_LeftRightArgs(lines=%s, higher=%s)' % (built, self.higher)

    def transpose(self):
        self._first_pointer_parent, self._second_pointer_parent = (self._second_pointer_parent, self._first_pointer_parent)
        self._first_pointer_index, self._second_pointer_index = (self._second_pointer_index, self._first_pointer_index)
        self._first_line_index, self._second_line_index = (self._second_line_index, self._first_line_index)
        return self

    @staticmethod
    def _build(expr):
        if isinstance(expr, ExprBuilder):
            return expr.build()
        if isinstance(expr, list):
            if len(expr) == 1:
                return expr[0]
            else:
                return expr[0](*[_LeftRightArgs._build(i) for i in expr[1]])
        else:
            return expr

    def build(self):
        data = [self._build(i) for i in self._lines]
        if self.higher != 1:
            data += [self._build(self.higher)]
        data = list(data)
        return data

    def matrix_form(self):
        if self.first != 1 and self.higher != 1:
            raise ValueError('higher dimensional array cannot be represented')

        def _get_shape(elem):
            if isinstance(elem, MatrixExpr):
                return elem.shape
            return (None, None)
        if _get_shape(self.first)[1] != _get_shape(self.second)[1]:
            if _get_shape(self.second) == (1, 1):
                return self.first * self.second[0, 0]
            if _get_shape(self.first) == (1, 1):
                return self.first[1, 1] * self.second.T
            raise ValueError('incompatible shapes')
        if self.first != 1:
            return self.first * self.second.T
        else:
            return self.higher

    def rank(self):
        """
        Number of dimensions different from trivial (warning: not related to
        matrix rank).
        """
        rank = 0
        if self.first != 1:
            rank += sum([i != 1 for i in self.first.shape])
        if self.second != 1:
            rank += sum([i != 1 for i in self.second.shape])
        if self.higher != 1:
            rank += 2
        return rank

    def _multiply_pointer(self, pointer, other):
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from ...tensor.array.expressions.array_expressions import ArrayContraction
        subexpr = ExprBuilder(ArrayContraction, [ExprBuilder(ArrayTensorProduct, [pointer, other]), (1, 2)], validator=ArrayContraction._validate)
        return subexpr

    def append_first(self, other):
        self.first_pointer *= other

    def append_second(self, other):
        self.second_pointer *= other