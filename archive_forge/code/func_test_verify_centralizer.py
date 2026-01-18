from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup,\
from sympy.combinatorics.testutil import _verify_bsgs, _cmp_perm_lists,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.random import shuffle
def test_verify_centralizer():
    S = SymmetricGroup(3)
    A = AlternatingGroup(3)
    triv = PermutationGroup([Permutation([0, 1, 2])])
    assert _verify_centralizer(S, S, centr=triv)
    assert _verify_centralizer(S, A, centr=A)