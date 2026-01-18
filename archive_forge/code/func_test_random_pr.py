from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def test_random_pr():
    D = DihedralGroup(6)
    r = 11
    n = 3
    _random_prec_n = {}
    _random_prec_n[0] = {'s': 7, 't': 3, 'x': 2, 'e': -1}
    _random_prec_n[1] = {'s': 5, 't': 5, 'x': 1, 'e': -1}
    _random_prec_n[2] = {'s': 3, 't': 4, 'x': 2, 'e': 1}
    D._random_pr_init(r, n, _random_prec_n=_random_prec_n)
    assert D._random_gens[11] == [0, 1, 2, 3, 4, 5]
    _random_prec = {'s': 2, 't': 9, 'x': 1, 'e': -1}
    assert D.random_pr(_random_prec=_random_prec) == Permutation([0, 5, 4, 3, 2, 1])