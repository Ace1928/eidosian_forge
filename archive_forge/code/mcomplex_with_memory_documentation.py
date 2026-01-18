from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import Mcomplex, VERBOSE, edge_and_arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron

        >>> M = McomplexWithMemory('zLALvwvMwLzzAQPQQkbcbeijmoomvwuvust'
        ...                        'wwytxtyxyahkswpmakguadppmrssxbkoxsi')
        >>> N = M.copy()
        >>> M.easy_simplify()
        True
        >>> len(M), len(M.move_memory)
        (1, 25)
        >>> M.rebuild(); M.isosig()
        'bkaagj'

        >>> N.invariant_tetrahedra = [N[1]]  # core solid torus
        >>> N.easy_simplify()
        True
        >>> len(N), len(N.move_memory)
        (15, 9)
        