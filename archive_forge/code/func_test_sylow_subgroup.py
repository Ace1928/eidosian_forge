from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_sylow_subgroup():
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5))
    S = P.sylow_subgroup(2)
    assert S.order() == 4
    P = DihedralGroup(12)
    S = P.sylow_subgroup(3)
    assert S.order() == 3
    P = PermutationGroup(Permutation(1, 5)(2, 4), Permutation(0, 1, 2, 3, 4, 5), Permutation(0, 2))
    S = P.sylow_subgroup(3)
    assert S.order() == 9
    S = P.sylow_subgroup(2)
    assert S.order() == 8
    P = SymmetricGroup(10)
    S = P.sylow_subgroup(2)
    assert S.order() == 256
    S = P.sylow_subgroup(3)
    assert S.order() == 81
    S = P.sylow_subgroup(5)
    assert S.order() == 25
    exp = 1
    length = 0
    for i in range(2, 9):
        P = SymmetricGroup(i)
        S = P.sylow_subgroup(2)
        ls = S.lower_central_series()
        if i // 2 ** exp > 0:
            assert len(ls) > length
            length = len(ls)
            exp += 1
        else:
            assert len(ls) == length
    G = SymmetricGroup(100)
    S = G.sylow_subgroup(3)
    assert G.order() % S.order() == 0
    assert G.order() / S.order() % 3 > 0
    G = AlternatingGroup(100)
    S = G.sylow_subgroup(2)
    assert G.order() % S.order() == 0
    assert G.order() / S.order() % 2 > 0
    G = DihedralGroup(18)
    S = G.sylow_subgroup(p=2)
    assert S.order() == 4
    G = DihedralGroup(50)
    S = G.sylow_subgroup(p=2)
    assert S.order() == 4