from .simplex import *
from .linalg import Matrix

    The boundary maps in the homology chain complex of the
    underlying cell-complex of a Mcomplex.

    >>> M = Mcomplex('o9_12345')
    >>> len(M.boundary_maps()) == 3
    True
    