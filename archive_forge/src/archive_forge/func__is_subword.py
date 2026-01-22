from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
def _is_subword(w):
    w, r = w.cyclic_reduction(removed=True)
    if r.is_identity or self.normal:
        return w in min_words
    else:
        t = [s[1] for s in min_words if isinstance(s, tuple) and s[0] == r]
        return [s for s in t if w.power_of(s)] != []