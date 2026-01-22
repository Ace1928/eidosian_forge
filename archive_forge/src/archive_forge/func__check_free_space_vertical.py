from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _check_free_space_vertical(dom_i, cod_i, dom_j, grid):
    """
        For a vertical morphism, checks whether there is free space
        (i.e., space not occupied by any objects) to the left of the
        morphism or to the right of it.
        """
    if dom_i < cod_i:
        start, end = (dom_i, cod_i)
        backwards = False
    else:
        start, end = (cod_i, dom_i)
        backwards = True
    if dom_j == 0:
        free_left = True
    else:
        free_left = not any((grid[i, dom_j - 1] for i in range(start, end + 1)))
    if dom_j == grid.width - 1:
        free_right = True
    else:
        free_right = not any((grid[i, dom_j + 1] for i in range(start, end + 1)))
    return (free_left, free_right, backwards)