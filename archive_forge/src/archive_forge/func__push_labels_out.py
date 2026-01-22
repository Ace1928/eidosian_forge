from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def _push_labels_out(self, morphisms_str_info, grid, object_coords):
    """
        For all straight morphisms which form the visual boundary of
        the laid out diagram, puts their labels on their outer sides.
        """

    def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
        """
            Given the information about room available to one side and
            to the other side of a morphism (``free1`` and ``free2``),
            sets the position of the morphism label in such a way that
            it is on the freer side.  This latter operations involves
            choice between ``pos1`` and ``pos2``, taking ``backwards``
            in consideration.

            Thus this function will do nothing if either both ``free1
            == True`` and ``free2 == True`` or both ``free1 == False``
            and ``free2 == False``.  In either case, choosing one side
            over the other presents no advantage.
            """
        if backwards:
            pos1, pos2 = (pos2, pos1)
        if free1 and (not free2):
            m_str_info.label_position = pos1
        elif free2 and (not free1):
            m_str_info.label_position = pos2
    for m, m_str_info in morphisms_str_info.items():
        if m_str_info.curving or m_str_info.forced_label_position:
            continue
        if m.domain == m.codomain:
            continue
        dom_i, dom_j = object_coords[m.domain]
        cod_i, cod_j = object_coords[m.codomain]
        if dom_i == cod_i:
            free_up, free_down, backwards = XypicDiagramDrawer._check_free_space_horizontal(dom_i, dom_j, cod_j, grid)
            set_label_position(free_up, free_down, '^', '_', backwards, m_str_info)
        elif dom_j == cod_j:
            free_left, free_right, backwards = XypicDiagramDrawer._check_free_space_vertical(dom_i, cod_i, dom_j, grid)
            set_label_position(free_left, free_right, '_', '^', backwards, m_str_info)
        else:
            free_up, free_down, backwards = XypicDiagramDrawer._check_free_space_diagonal(dom_i, cod_i, dom_j, cod_j, grid)
            set_label_position(free_up, free_down, '^', '_', backwards, m_str_info)