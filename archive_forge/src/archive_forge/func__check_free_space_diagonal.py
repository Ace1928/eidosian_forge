from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid):
    """
        For a diagonal morphism, checks whether there is free space
        (i.e., space not occupied by any objects) above the morphism
        or below it.
        """

    def abs_xrange(start, end):
        if start < end:
            return range(start, end + 1)
        else:
            return range(end, start + 1)
    if dom_i < cod_i and dom_j < cod_j:
        start_i, start_j = (dom_i, dom_j)
        end_i, end_j = (cod_i, cod_j)
        backwards = False
    elif dom_i > cod_i and dom_j > cod_j:
        start_i, start_j = (cod_i, cod_j)
        end_i, end_j = (dom_i, dom_j)
        backwards = True
    if dom_i < cod_i and dom_j > cod_j:
        start_i, start_j = (dom_i, dom_j)
        end_i, end_j = (cod_i, cod_j)
        backwards = True
    elif dom_i > cod_i and dom_j < cod_j:
        start_i, start_j = (cod_i, cod_j)
        end_i, end_j = (dom_i, dom_j)
        backwards = False
    alpha = float(end_i - start_i) / (end_j - start_j)
    free_up = True
    free_down = True
    for i in abs_xrange(start_i, end_i):
        if not free_up and (not free_down):
            break
        for j in abs_xrange(start_j, end_j):
            if not free_up and (not free_down):
                break
            if (i, j) == (start_i, start_j):
                continue
            if j == start_j:
                alpha1 = 'inf'
            else:
                alpha1 = float(i - start_i) / (j - start_j)
            if grid[i, j]:
                if alpha1 == 'inf' or abs(alpha1) > abs(alpha):
                    free_down = False
                elif abs(alpha1) < abs(alpha):
                    free_up = False
    return (free_up, free_down, backwards)