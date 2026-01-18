from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_transitivity_degree():
    perm = Permutation([1, 2, 0])
    C = PermutationGroup([perm])
    assert C.transitivity_degree == 1
    gen1 = Permutation([1, 2, 0, 3, 4])
    gen2 = Permutation([1, 2, 3, 4, 0])
    Alt = PermutationGroup([gen1, gen2])
    assert Alt.transitivity_degree == 3