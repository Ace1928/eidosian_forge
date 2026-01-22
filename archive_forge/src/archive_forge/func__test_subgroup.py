from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
from sympy.combinatorics.free_groups import (free_group, FreeGroup)
from sympy.testing.pytest import slow
def _test_subgroup(K, T, S):
    _gens = T(K.generators)
    assert all((elem in S for elem in _gens))
    assert T.is_injective()
    assert T.image().order() == S.order()