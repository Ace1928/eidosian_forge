from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io

        Return the list of isomorphisms between the MComplexes M and N.
        If `at_most_one` is `True`, only returns the first one found (but
        still as a list).

        >>> tri_data = [([0,1,0,1], [(2,1,0,3), (0,3,2,1), (2,1,0,3), (0,1,3,2)]),
        ...             ([1,1,0,0], [(1,0,2,3), (1,0,2,3), (0,1,3,2), (0,3,2,1)])]
        >>> M = Mcomplex(tri_data)
        >>> N = Mcomplex(M.isosig())
        >>> isos = M.isomorphisms_to(N); len(isos)
        4
        >>> isos[0]
        {0: [tet0, (0, 2, 1, 3)], 1: [tet1, (0, 2, 1, 3)]}
        >>> len(M.isomorphisms_to(N, orientation_preserving=True))
        2
        >>> M.two_to_three(Arrow(E01, F3, M[0])); M.rebuild()
        True
        >>> len(M), len(N)
        (3, 2)
        >>> M.isomorphisms_to(N)
        []
        >>> F = Mcomplex('m004')
        >>> N.isomorphisms_to(F)
        []
        >>> N = Mcomplex(M.isosig())
        >>> M.isomorphisms_to(N, at_most_one=True)[0]
        {0: [tet1, (0, 2, 3, 1)], 1: [tet2, (0, 2, 3, 1)], 2: [tet0, (0, 3, 1, 2)]}
        >>> M = Mcomplex(tri_data)
        >>> M.two_to_three(Arrow(E01, F3, M[0])); M.two_to_three(Arrow(E01, F3, M[1]))
        True
        True
        >>> M.rebuild()
        >>> len(M) == 4
        True
        >>> N = Mcomplex(M.isosig())
        >>> M.isomorphisms_to(N, at_most_one=True)[0]  # doctest: +NORMALIZE_WHITESPACE
        {0: [tet0, (1, 3, 0, 2)], 1: [tet1, (3, 0, 1, 2)],
         2: [tet3, (2, 0, 3, 1)], 3: [tet2, (3, 1, 2, 0)]}
        