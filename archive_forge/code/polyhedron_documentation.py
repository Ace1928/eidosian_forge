from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int
Return corners to their original positions.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron as T
        >>> T = T.copy()
        >>> T.corners
        (0, 1, 2, 3)
        >>> T.rotate(0)
        >>> T.corners
        (0, 2, 3, 1)
        >>> T.reset()
        >>> T.corners
        (0, 1, 2, 3)
        