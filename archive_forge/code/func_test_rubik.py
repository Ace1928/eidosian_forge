from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
@XFAIL
def test_rubik():
    skip('takes too much time')
    G = PermutationGroup(rubik_cube_generators())
    assert G.order() == 43252003274489856000
    G1 = PermutationGroup(G[:3])
    assert G1.order() == 170659735142400
    assert not G1.is_normal(G)
    G2 = G.normal_closure(G1.generators)
    assert G2.is_subgroup(G)