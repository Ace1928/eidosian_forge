from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_stabilizer():
    S = SymmetricGroup(2)
    H = S.stabilizer(0)
    assert H.generators == [Permutation(1)]
    a = Permutation([2, 0, 1, 3, 4, 5])
    b = Permutation([2, 1, 3, 4, 5, 0])
    G = PermutationGroup([a, b])
    G0 = G.stabilizer(0)
    assert G0.order() == 60
    gens_cube = [[1, 3, 5, 7, 0, 2, 4, 6], [1, 3, 0, 2, 5, 7, 4, 6]]
    gens = [Permutation(p) for p in gens_cube]
    G = PermutationGroup(gens)
    G2 = G.stabilizer(2)
    assert G2.order() == 6
    G2_1 = G2.stabilizer(1)
    v = list(G2_1.generate(af=True))
    assert v == [[0, 1, 2, 3, 4, 5, 6, 7], [3, 1, 2, 0, 7, 5, 6, 4]]
    gens = ((1, 2, 0, 4, 5, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), (0, 1, 2, 3, 4, 5, 19, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 7, 17, 18), (0, 1, 2, 3, 4, 5, 6, 7, 9, 18, 16, 11, 12, 13, 14, 15, 8, 17, 10, 19))
    gens = [Permutation(p) for p in gens]
    G = PermutationGroup(gens)
    G2 = G.stabilizer(2)
    assert G2.order() == 181440
    S = SymmetricGroup(3)
    assert [G.order() for G in S.basic_stabilizers] == [6, 2]