from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _drop_redundant_triangles(triangles, skeleton):
    """
        Returns a list which contains only those triangles who have
        morphisms associated with at least two edges.
        """
    return [tri for tri in triangles if len([e for e in tri if skeleton[e]]) >= 2]