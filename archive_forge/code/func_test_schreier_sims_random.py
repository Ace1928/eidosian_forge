from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_schreier_sims_random():
    assert sorted(Tetra.pgroup.base) == [0, 1]
    S = SymmetricGroup(3)
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([1, 0, 2]), Permutation([0, 2, 1])]
    assert S.schreier_sims_random(base, strong_gens, 5) == (base, strong_gens)
    D = DihedralGroup(3)
    _random_prec = {'g': [Permutation([2, 0, 1]), Permutation([1, 2, 0]), Permutation([1, 0, 2])]}
    base = [0, 1]
    strong_gens = [Permutation([1, 2, 0]), Permutation([2, 1, 0]), Permutation([0, 2, 1])]
    assert D.schreier_sims_random([], D.generators, 2, _random_prec=_random_prec) == (base, strong_gens)