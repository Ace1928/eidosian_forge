from sympy.core.containers import Tuple
from sympy.combinatorics.generators import rubik_cube_generators
from sympy.combinatorics.homomorphisms import is_isomorphic
from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup,\
from sympy.combinatorics.perm_groups import (PermutationGroup,
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.polyhedron import tetrahedron as Tetra, cube
from sympy.combinatorics.testutil import _verify_bsgs, _verify_centralizer,\
from sympy.testing.pytest import skip, XFAIL, slow
def _subgroup_search(i, j, k):
    prop_true = lambda x: True
    prop_fix_points = lambda x: [x(point) for point in points] == points
    prop_comm_g = lambda x: rmul(x, g) == rmul(g, x)
    prop_even = lambda x: x.is_even
    for i in range(i, j, k):
        S = SymmetricGroup(i)
        A = AlternatingGroup(i)
        C = CyclicGroup(i)
        Sym = S.subgroup_search(prop_true)
        assert Sym.is_subgroup(S)
        Alt = S.subgroup_search(prop_even)
        assert Alt.is_subgroup(A)
        Sym = S.subgroup_search(prop_true, init_subgroup=C)
        assert Sym.is_subgroup(S)
        points = [7]
        assert S.stabilizer(7).is_subgroup(S.subgroup_search(prop_fix_points))
        points = [3, 4]
        assert S.stabilizer(3).stabilizer(4).is_subgroup(S.subgroup_search(prop_fix_points))
        points = [3, 5]
        fix35 = A.subgroup_search(prop_fix_points)
        points = [5]
        fix5 = A.subgroup_search(prop_fix_points)
        assert A.subgroup_search(prop_fix_points, init_subgroup=fix35).is_subgroup(fix5)
        base, strong_gens = A.schreier_sims_incremental()
        g = A.generators[0]
        comm_g = A.subgroup_search(prop_comm_g, base=base, strong_gens=strong_gens)
        assert _verify_bsgs(comm_g, base, comm_g.generators) is True
        assert [prop_comm_g(gen) is True for gen in comm_g.generators]