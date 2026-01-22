from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _choose_target_cell(pt1, pt2, edge, obj, skeleton, grid):
    """
        Given two points, ``pt1`` and ``pt2``, and the welding edge
        ``edge``, chooses one of the two points to place the opposing
        vertex ``obj`` of the triangle.  If neither of this points
        fits, returns ``None``.
        """
    pt1_empty = DiagramGrid._empty_point(pt1, grid)
    pt2_empty = DiagramGrid._empty_point(pt2, grid)
    if pt1_empty and pt2_empty:
        A = grid[edge[0]]
        if skeleton.get(frozenset([A, obj])):
            return pt1
        else:
            return pt2
    if pt1_empty:
        return pt1
    elif pt2_empty:
        return pt2
    else:
        return None