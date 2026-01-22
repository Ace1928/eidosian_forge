from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _morphism_sort_key(morphism, object_coords):
    """
        Provides a morphism sorting key such that horizontal or
        vertical morphisms between neighbouring objects come
        first, then horizontal or vertical morphisms between more
        far away objects, and finally, all other morphisms.
        """
    i, j = object_coords[morphism.domain]
    target_i, target_j = object_coords[morphism.codomain]
    if morphism.domain == morphism.codomain:
        return (3, 0, default_sort_key(morphism))
    if target_i == i:
        return (1, abs(target_j - j), default_sort_key(morphism))
    if target_j == j:
        return (1, abs(target_i - i), default_sort_key(morphism))
    return (2, 0, default_sort_key(morphism))