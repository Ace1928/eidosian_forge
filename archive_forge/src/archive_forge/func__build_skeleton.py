from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _build_skeleton(morphisms):
    """
        Creates a dictionary which maps edges to corresponding
        morphisms.  Thus for a morphism `f:A\rightarrow B`, the edge
        `(A, B)` will be associated with `f`.  This function also adds
        to the list those edges which are formed by juxtaposition of
        two edges already in the list.  These new edges are not
        associated with any morphism and are only added to assure that
        the diagram can be decomposed into triangles.
        """
    edges = {}
    for morphism in morphisms:
        DiagramGrid._add_edge_append(edges, frozenset([morphism.domain, morphism.codomain]), morphism)
    edges1 = dict(edges)
    for w in edges1:
        for v in edges1:
            wv = DiagramGrid._juxtapose_edges(w, v)
            if wv and wv not in edges:
                edges[wv] = []
    return edges