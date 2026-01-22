from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _get_undirected_graph(objects, merged_morphisms):
    """
        Given the objects and the relevant morphisms of a diagram,
        returns the adjacency lists of the underlying undirected
        graph.
        """
    adjlists = {}
    for obj in objects:
        adjlists[obj] = []
    for morphism in merged_morphisms:
        adjlists[morphism.domain].append(morphism.codomain)
        adjlists[morphism.codomain].append(morphism.domain)
    for obj in adjlists.keys():
        adjlists[obj].sort(key=default_sort_key)
    return adjlists