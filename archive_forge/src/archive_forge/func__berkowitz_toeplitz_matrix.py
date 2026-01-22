from types import FunctionType
from sympy.core.numbers import Float, Integer
from sympy.core.singleton import S
from sympy.core.symbol import uniquely_named_symbol
from sympy.core.mul import Mul
from sympy.polys import PurePoly, cancel
from sympy.functions.combinatorial.numbers import nC
from sympy.polys.matrices.domainmatrix import DomainMatrix
from .common import NonSquareMatrixError
from .utilities import (
def _berkowitz_toeplitz_matrix(M):
    """Return (A,T) where T the Toeplitz matrix used in the Berkowitz algorithm
    corresponding to ``M`` and A is the first principal submatrix.
    """
    if M.rows == 0 and M.cols == 0:
        return M._new(1, 1, [M.one])
    a, R = (M[0, 0], M[0, 1:])
    C, A = (M[1:, 0], M[1:, 1:])
    diags = [C]
    for i in range(M.rows - 2):
        diags.append(A.multiply(diags[i], dotprodsimp=None))
    diags = [(-R).multiply(d, dotprodsimp=None)[0, 0] for d in diags]
    diags = [M.one, -a] + diags

    def entry(i, j):
        if j > i:
            return M.zero
        return diags[i - j]
    toeplitz = M._new(M.cols + 1, M.rows, entry)
    return (A, toeplitz)