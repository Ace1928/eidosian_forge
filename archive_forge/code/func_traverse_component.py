from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def traverse_component(object, current_index):
    """
            Does a depth-first search traversal of the component
            containing ``object``.
            """
    component_index[object] = current_index
    for o in adjlist[object]:
        if component_index[o] is None:
            traverse_component(o, current_index)