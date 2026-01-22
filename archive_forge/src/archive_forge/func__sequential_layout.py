from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _sequential_layout(diagram, merged_morphisms):
    """
        Lays out the diagram in "sequential" layout.  This method
        will attempt to produce a result as close to a line as
        possible.  For linear diagrams, the result will actually be a
        line.
        """
    objects = diagram.objects
    sorted_objects = sorted(objects, key=default_sort_key)
    adjlists = DiagramGrid._get_undirected_graph(objects, merged_morphisms)
    root = sorted_objects[0]
    mindegree = len(adjlists[root])
    for obj in sorted_objects:
        current_degree = len(adjlists[obj])
        if current_degree < mindegree:
            root = obj
            mindegree = current_degree
    grid = _GrowableGrid(1, 1)
    grid[0, 0] = root
    placed_objects = {root}

    def place_objects(pt, placed_objects):
        """
            Does depth-first search in the underlying graph of the
            diagram and places the objects en route.
            """
        new_pt = (pt[0], pt[1] + 1)
        for adjacent_obj in adjlists[grid[pt]]:
            if adjacent_obj in placed_objects:
                continue
            DiagramGrid._put_object(new_pt, adjacent_obj, grid, [])
            placed_objects.add(adjacent_obj)
            placed_objects.update(place_objects(new_pt, placed_objects))
            new_pt = (new_pt[0] + 1, new_pt[1])
        return placed_objects
    place_objects((0, 0), placed_objects)
    return grid