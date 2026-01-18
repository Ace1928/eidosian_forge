from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup,\
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.util import _check_cycles_alt_sym, _strip,\
from sympy.combinatorics.testutil import _verify_bsgs
def test_strong_gens_from_distr():
    strong_gens_distr = [[Permutation([0, 2, 1]), Permutation([1, 2, 0]), Permutation([1, 0, 2])], [Permutation([0, 2, 1])]]
    assert _strong_gens_from_distr(strong_gens_distr) == [Permutation([0, 2, 1]), Permutation([1, 2, 0]), Permutation([1, 0, 2])]