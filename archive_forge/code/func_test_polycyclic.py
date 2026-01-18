from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_polycyclic():
    a = Permutation([0, 1, 2])
    b = Permutation([2, 1, 0])
    G = PermutationGroup([a, b])
    assert G.is_polycyclic is True
    a = Permutation([1, 2, 3, 4, 0])
    b = Permutation([1, 0, 2, 3, 4])
    G = PermutationGroup([a, b])
    assert G.is_polycyclic is False