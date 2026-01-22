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
def _det_berkowitz(M):
    """ Use the Berkowitz algorithm to compute the determinant."""
    if not M.is_square:
        raise NonSquareMatrixError()
    if M.rows == 0:
        return M.one
    berk_vector = _berkowitz_vector(M)
    return (-1) ** (len(berk_vector) - 1) * berk_vector[-1]