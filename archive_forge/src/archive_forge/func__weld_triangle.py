from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _weld_triangle(tri, welding_edge, fringe, grid, skeleton):
    """
        If possible, welds the triangle ``tri`` to ``fringe`` and
        returns ``False``.  If this method encounters a degenerate
        situation in the fringe and corrects it such that a restart of
        the search is required, it returns ``True`` (which means that
        a restart in finding triangle weldings is required).

        A degenerate situation is a situation when an edge listed in
        the fringe does not belong to the visual boundary of the
        diagram.
        """
    a, b = welding_edge
    target_cell = None
    obj = DiagramGrid._other_vertex(tri, (grid[a], grid[b]))
    if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1:
        target_cell = (a[0], b[1])
        if grid[target_cell]:
            target_cell = (b[0], a[1])
            if grid[target_cell]:
                fringe.remove((a, b))
                return True
    elif a[0] == b[0]:
        down_left = (a[0] + 1, a[1])
        down_right = (a[0] + 1, b[1])
        target_cell = DiagramGrid._choose_target_cell(down_left, down_right, (a, b), obj, skeleton, grid)
        if not target_cell:
            up_left = (a[0] - 1, a[1])
            up_right = (a[0] - 1, b[1])
            target_cell = DiagramGrid._choose_target_cell(up_left, up_right, (a, b), obj, skeleton, grid)
            if not target_cell:
                fringe.remove((a, b))
                return True
    elif a[1] == b[1]:
        right_up = (a[0], a[1] + 1)
        right_down = (b[0], a[1] + 1)
        target_cell = DiagramGrid._choose_target_cell(right_up, right_down, (a, b), obj, skeleton, grid)
        if not target_cell:
            left_up = (a[0], a[1] - 1)
            left_down = (b[0], a[1] - 1)
            target_cell = DiagramGrid._choose_target_cell(left_up, left_down, (a, b), obj, skeleton, grid)
            if not target_cell:
                fringe.remove((a, b))
                return True
    offset = DiagramGrid._put_object(target_cell, obj, grid, fringe)
    target_cell = (target_cell[0] + offset[0], target_cell[1] + offset[1])
    a = (a[0] + offset[0], a[1] + offset[1])
    b = (b[0] + offset[0], b[1] + offset[1])
    fringe.extend([(a, target_cell), (b, target_cell)])
    return False