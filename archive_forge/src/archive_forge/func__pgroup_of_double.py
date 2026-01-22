from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int
def _pgroup_of_double(polyh, ordered_faces, pgroup):
    n = len(ordered_faces[0])
    fmap = dict(zip(ordered_faces, range(len(ordered_faces))))
    flat_faces = flatten(ordered_faces)
    new_pgroup = []
    for i, p in enumerate(pgroup):
        h = polyh.copy()
        h.rotate(p)
        c = h.corners
        reorder = unflatten([c[j] for j in flat_faces], n)
        reorder = [tuple(map(as_int, minlex(f, directed=False))) for f in reorder]
        new_pgroup.append(Perm([fmap[f] for f in reorder]))
    return new_pgroup