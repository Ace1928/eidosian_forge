from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup,\
from sympy.combinatorics.testutil import _verify_bsgs, _cmp_perm_lists,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.random import shuffle
def test_verify_bsgs():
    S = SymmetricGroup(5)
    S.schreier_sims()
    base = S.base
    strong_gens = S.strong_gens
    assert _verify_bsgs(S, base, strong_gens) is True
    assert _verify_bsgs(S, base[:-1], strong_gens) is False
    assert _verify_bsgs(S, base, S.generators) is False