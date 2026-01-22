from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify
def _is_positive_definite_GE(M):
    """A division-free gaussian elimination method for testing
    positive-definiteness."""
    M = M.as_mutable()
    size = M.rows
    for i in range(size):
        is_positive = M[i, i].is_positive
        if is_positive is not True:
            return is_positive
        for j in range(i + 1, size):
            M[j, i + 1:] = M[i, i] * M[j, i + 1:] - M[j, i] * M[i, i + 1:]
    return True