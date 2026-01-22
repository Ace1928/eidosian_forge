from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _get_connected_components(objects, merged_morphisms):
    """
        Given a container of morphisms, returns a list of connected
        components formed by these morphisms.  A connected component
        is represented by a diagram consisting of the corresponding
        morphisms.
        """
    component_index = {}
    for o in objects:
        component_index[o] = None
    adjlist = DiagramGrid._get_undirected_graph(objects, merged_morphisms)

    def traverse_component(object, current_index):
        """
            Does a depth-first search traversal of the component
            containing ``object``.
            """
        component_index[object] = current_index
        for o in adjlist[object]:
            if component_index[o] is None:
                traverse_component(o, current_index)
    current_index = 0
    for o in adjlist:
        if component_index[o] is None:
            traverse_component(o, current_index)
            current_index += 1
    component_objects = [[] for i in range(current_index)]
    for o, idx in component_index.items():
        component_objects[idx].append(o)
    component_morphisms = []
    for component in component_objects:
        current_morphisms = {}
        for m in merged_morphisms:
            if m.domain in component and m.codomain in component:
                current_morphisms[m] = merged_morphisms[m]
        if len(component) == 1:
            current_morphisms[IdentityMorphism(component[0])] = FiniteSet()
        component_morphisms.append(Diagram(current_morphisms))
    return component_morphisms