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
def _perm_group_list(self, method_name, *args):
    """
        Given the name of a `PermutationGroup` method (returning a subgroup
        or a list of subgroups) and (optionally) additional arguments it takes,
        return a list or a list of lists containing the generators of this (or
        these) subgroups in terms of the generators of `self`.

        """
    P, T = self._to_perm_group()
    perm_result = getattr(P, method_name)(*args)
    single = False
    if isinstance(perm_result, PermutationGroup):
        perm_result, single = ([perm_result], True)
    result = []
    for group in perm_result:
        gens = group.generators
        result.append(T.invert(gens))
    return result[0] if single else result