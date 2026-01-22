from itertools import chain
from .exceptions import DMBadInputError, DMShapeError, DMDomainError
from .dense import (
from sympy.polys.domains import QQ
from .lll import ddm_lll, ddm_lll_transform
from .sdm import SDM

        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        