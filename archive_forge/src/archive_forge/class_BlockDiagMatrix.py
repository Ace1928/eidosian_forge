from sympy.assumptions.ask import (Q, ask)
from sympy.core import Basic, Add, Mul, S
from sympy.core.sympify import _sympify
from sympy.functions import adjoint
from sympy.functions.elementary.complexes import re, im
from sympy.strategies import typed, exhaust, condition, do_one, unpack
from sympy.strategies.traverse import bottom_up
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent
from sympy.matrices import Matrix, ShapeError
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.expressions.determinant import det, Determinant
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.matrices.expressions.trace import trace
from sympy.matrices.expressions.transpose import Transpose, transpose
class BlockDiagMatrix(BlockMatrix):
    """A sparse matrix with block matrices along its diagonals

    Examples
    ========

    >>> from sympy import MatrixSymbol, BlockDiagMatrix, symbols
    >>> n, m, l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> BlockDiagMatrix(X, Y)
    Matrix([
    [X, 0],
    [0, Y]])

    Notes
    =====

    If you want to get the individual diagonal blocks, use
    :meth:`get_diag_blocks`.

    See Also
    ========

    sympy.matrices.dense.diag
    """

    def __new__(cls, *mats):
        return Basic.__new__(BlockDiagMatrix, *[_sympify(m) for m in mats])

    @property
    def diag(self):
        return self.args

    @property
    def blocks(self):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        mats = self.args
        data = [[mats[i] if i == j else ZeroMatrix(mats[i].rows, mats[j].cols) for j in range(len(mats))] for i in range(len(mats))]
        return ImmutableDenseMatrix(data, evaluate=False)

    @property
    def shape(self):
        return (sum((block.rows for block in self.args)), sum((block.cols for block in self.args)))

    @property
    def blockshape(self):
        n = len(self.args)
        return (n, n)

    @property
    def rowblocksizes(self):
        return [block.rows for block in self.args]

    @property
    def colblocksizes(self):
        return [block.cols for block in self.args]

    def _all_square_blocks(self):
        """Returns true if all blocks are square"""
        return all((mat.is_square for mat in self.args))

    def _eval_determinant(self):
        if self._all_square_blocks():
            return Mul(*[det(mat) for mat in self.args])
        return S.Zero

    def _eval_inverse(self, expand='ignored'):
        if self._all_square_blocks():
            return BlockDiagMatrix(*[mat.inverse() for mat in self.args])
        raise NonInvertibleMatrixError('Matrix det == 0; not invertible.')

    def _eval_transpose(self):
        return BlockDiagMatrix(*[mat.transpose() for mat in self.args])

    def _blockmul(self, other):
        if isinstance(other, BlockDiagMatrix) and self.colblocksizes == other.rowblocksizes:
            return BlockDiagMatrix(*[a * b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockmul(self, other)

    def _blockadd(self, other):
        if isinstance(other, BlockDiagMatrix) and self.blockshape == other.blockshape and (self.rowblocksizes == other.rowblocksizes) and (self.colblocksizes == other.colblocksizes):
            return BlockDiagMatrix(*[a + b for a, b in zip(self.args, other.args)])
        else:
            return BlockMatrix._blockadd(self, other)

    def get_diag_blocks(self):
        """Return the list of diagonal blocks of the matrix.

        Examples
        ========

        >>> from sympy import BlockDiagMatrix, Matrix

        >>> A = Matrix([[1, 2], [3, 4]])
        >>> B = Matrix([[5, 6], [7, 8]])
        >>> M = BlockDiagMatrix(A, B)

        How to get diagonal blocks from the block diagonal matrix:

        >>> diag_blocks = M.get_diag_blocks()
        >>> diag_blocks[0]
        Matrix([
        [1, 2],
        [3, 4]])
        >>> diag_blocks[1]
        Matrix([
        [5, 6],
        [7, 8]])
        """
        return self.args