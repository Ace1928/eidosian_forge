from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int
def _string_to_perm(s):
    rv = [Perm(range(20))]
    p = None
    for si in s:
        if si not in '01':
            count = int(si) - 1
        else:
            count = 1
            if si == '0':
                p = _f0
            elif si == '1':
                p = _f1
        rv.extend([p] * count)
    return Perm.rmul(*rv)