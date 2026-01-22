from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _process_horizontal_morphism(i, j, target_j, grid, morphisms_str_info, object_coords):
    """
        Produces the information required for constructing the string
        representation of a horizontal morphism.  This function is
        invoked from ``_process_morphism``.

        See Also
        ========

        _process_morphism
        """
    backwards = False
    start = j
    end = target_j
    if end < start:
        start, end = (end, start)
        backwards = True
    up = []
    down = []
    straight_horizontal = []
    for k in range(start + 1, end):
        obj = grid[i, k]
        if not obj:
            continue
        for m in morphisms_str_info:
            if m.domain == obj:
                end_i, end_j = object_coords[m.codomain]
            elif m.codomain == obj:
                end_i, end_j = object_coords[m.domain]
            else:
                continue
            if end_i > i:
                down.append(m)
            elif end_i < i:
                up.append(m)
            elif not morphisms_str_info[m].curving:
                straight_horizontal.append(m)
    if len(up) < len(down):
        if backwards:
            curving = '_'
            label_pos = '_'
        else:
            curving = '^'
            label_pos = '^'
        for m in straight_horizontal:
            i1, j1 = object_coords[m.domain]
            i2, j2 = object_coords[m.codomain]
            m_str_info = morphisms_str_info[m]
            if j1 < j2:
                m_str_info.label_position = '_'
            else:
                m_str_info.label_position = '^'
            m_str_info.forced_label_position = True
    else:
        if backwards:
            curving = '^'
            label_pos = '^'
        else:
            curving = '_'
            label_pos = '_'
        for m in straight_horizontal:
            i1, j1 = object_coords[m.domain]
            i2, j2 = object_coords[m.codomain]
            m_str_info = morphisms_str_info[m]
            if j1 < j2:
                m_str_info.label_position = '^'
            else:
                m_str_info.label_position = '_'
            m_str_info.forced_label_position = True
    return (curving, label_pos)