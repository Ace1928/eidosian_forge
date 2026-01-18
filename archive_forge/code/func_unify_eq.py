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
def unify_eq(A, B):
    if A.shape != B.shape:
        return False
    if A.domain != B.domain:
        A, B = A.unify(B)
    return A == B