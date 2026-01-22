from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _find_triangle_to_weld(triangles, fringe, grid):
    """
        Finds, if possible, a triangle and an edge in the ``fringe`` to
        which the triangle could be attached.  Returns the tuple
        containing the triangle and the index of the corresponding
        edge in the ``fringe``.

        This function relies on the fact that objects are unique in
        the diagram.
        """
    for triangle in triangles:
        for a, b in fringe:
            if frozenset([grid[a], grid[b]]) in triangle:
                return (triangle, (a, b))
    return None