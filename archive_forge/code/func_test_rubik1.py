from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_rubik1():
    gens = rubik_cube_generators()
    gens1 = [gens[-1]] + [p ** 2 for p in gens[1:]]
    G1 = PermutationGroup(gens1)
    assert G1.order() == 19508428800
    gens2 = [p ** 2 for p in gens]
    G2 = PermutationGroup(gens2)
    assert G2.order() == 663552
    assert G2.is_subgroup(G1, 0)
    C1 = G1.derived_subgroup()
    assert C1.order() == 4877107200
    assert C1.is_subgroup(G1, 0)
    assert not G2.is_subgroup(C1, 0)
    G = RubikGroup(2)
    assert G.order() == 3674160