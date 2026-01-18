from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def transversal2coset(size, base, transversal):
    a = []
    j = 0
    for i in range(size):
        if i in base:
            a.append(sorted(transversal[j].values()))
            j += 1
        else:
            a.append([list(range(size))])
    j = len(a) - 1
    while a[j] == [list(range(size))]:
        j -= 1
    return a[:j + 1]