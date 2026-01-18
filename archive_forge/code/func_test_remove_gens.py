from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
from sympy.combinatorics.testutil import _verify_bsgs
def test_remove_gens():
    S = SymmetricGroup(10)
    base, strong_gens = S.schreier_sims_incremental()
    new_gens = _remove_gens(base, strong_gens)
    assert _verify_bsgs(S, base, new_gens) is True
    A = AlternatingGroup(7)
    base, strong_gens = A.schreier_sims_incremental()
    new_gens = _remove_gens(base, strong_gens)
    assert _verify_bsgs(A, base, new_gens) is True
    D = DihedralGroup(2)
    base, strong_gens = D.schreier_sims_incremental()
    new_gens = _remove_gens(base, strong_gens)
    assert _verify_bsgs(D, base, new_gens) is True