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
def _eigenvects_sympy(M, iszerofunc, simplify=True, **flags):
    eigenvals = M.eigenvals(rational=False, **flags)
    for x in eigenvals:
        if x.has(CRootOf):
            raise MatrixError('Eigenvector computation is not implemented if the matrix have eigenvalues in CRootOf form')
    eigenvals = sorted(eigenvals.items(), key=default_sort_key)
    ret = []
    for val, mult in eigenvals:
        vects = _eigenspace(M, val, iszerofunc=iszerofunc, simplify=simplify)
        ret.append((val, mult, vects))
    return ret