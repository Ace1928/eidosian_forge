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
def _finite_index_subgroup(self, s=None):
    """
        Find the elements of `self` that generate a finite index subgroup
        and, if found, return the list of elements and the coset table of `self` by
        the subgroup, otherwise return `(None, None)`

        """
    gen = self.most_frequent_generator()
    rels = list(self.generators)
    rels.extend(self.relators)
    if not s:
        if len(self.generators) == 2:
            s = [gen] + [g for g in self.generators if g != gen]
        else:
            rand = self.free_group.identity
            i = 0
            while (rand in rels or rand ** (-1) in rels or rand.is_identity) and i < 10:
                rand = self.random()
                i += 1
            s = [gen, rand] + [g for g in self.generators if g != gen]
    mid = (len(s) + 1) // 2
    half1 = s[:mid]
    half2 = s[mid:]
    draft1 = None
    draft2 = None
    m = 200
    C = None
    while not C and m / 2 < CosetTable.coset_table_max_limit:
        m = min(m, CosetTable.coset_table_max_limit)
        draft1 = self.coset_enumeration(half1, max_cosets=m, draft=draft1, incomplete=True)
        if draft1.is_complete():
            C = draft1
            half = half1
        else:
            draft2 = self.coset_enumeration(half2, max_cosets=m, draft=draft2, incomplete=True)
            if draft2.is_complete():
                C = draft2
                half = half2
        if not C:
            m *= 2
    if not C:
        return (None, None)
    C.compress()
    return (half, C)