from functools import reduce
from typing import Union as tUnion, Tuple as tTuple
from sympy.core.sympify import _sympify
from ..domains import Domain
from ..constructor import construct_domain
from .exceptions import (DMNonSquareMatrixError, DMShapeError,
from .ddm import DDM
from .sdm import SDM
from .domainscalar import DomainScalar
from sympy.polys.domains import ZZ, EXRAW, QQ
@classmethod
def _unify_fmt(cls, *matrices, fmt=None):
    """Convert matrices to the same format.

        If all matrices have the same format, then return unmodified.
        Otherwise convert both to the preferred format given as *fmt* which
        should be 'dense' or 'sparse'.
        """
    formats = {matrix.rep.fmt for matrix in matrices}
    if len(formats) == 1:
        return matrices
    if fmt == 'sparse':
        return tuple((matrix.to_sparse() for matrix in matrices))
    elif fmt == 'dense':
        return tuple((matrix.to_dense() for matrix in matrices))
    else:
        raise ValueError("fmt should be 'sparse' or 'dense'")