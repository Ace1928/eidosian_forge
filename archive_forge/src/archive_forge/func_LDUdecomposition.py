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
def LDUdecomposition(self):
    """Returns the Block LDU decomposition of
        a 2x2 Block Matrix

        Returns
        =======

        (L, D, U) : Matrices
            L : Lower Diagonal Matrix
            D : Diagonal Matrix
            U : Upper Diagonal Matrix

        Examples
        ========

        >>> from sympy import symbols, MatrixSymbol, BlockMatrix, block_collapse
        >>> m, n = symbols('m n')
        >>> A = MatrixSymbol('A', n, n)
        >>> B = MatrixSymbol('B', n, m)
        >>> C = MatrixSymbol('C', m, n)
        >>> D = MatrixSymbol('D', m, m)
        >>> X = BlockMatrix([[A, B], [C, D]])
        >>> L, D, U = X.LDUdecomposition()
        >>> block_collapse(L*D*U)
        Matrix([
        [A, B],
        [C, D]])

        Raises
        ======

        ShapeError
            If the block matrix is not a 2x2 matrix

        NonInvertibleMatrixError
            If the matrix "A" is non-invertible

        See Also
        ========
        sympy.matrices.expressions.blockmatrix.BlockMatrix.UDLdecomposition
        sympy.matrices.expressions.blockmatrix.BlockMatrix.LUdecomposition
        """
    if self.blockshape == (2, 2):
        [[A, B], [C, D]] = self.blocks.tolist()
        try:
            AI = A.I
        except NonInvertibleMatrixError:
            raise NonInvertibleMatrixError('Block LDU decomposition cannot be calculated when                    "A" is singular')
        Ip = Identity(B.shape[0])
        Iq = Identity(B.shape[1])
        Z = ZeroMatrix(*B.shape)
        L = BlockMatrix([[Ip, Z], [C * AI, Iq]])
        D = BlockDiagMatrix(A, self.schur())
        U = BlockMatrix([[Ip, AI * B], [Z.T, Iq]])
        return (L, D, U)
    else:
        raise ShapeError('Block LDU decomposition is supported only for 2x2 block matrices')