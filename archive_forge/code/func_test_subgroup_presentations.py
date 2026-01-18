from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
from sympy.combinatorics.free_groups import (free_group, FreeGroup)
from sympy.testing.pytest import slow
def test_subgroup_presentations():
    F, x, y = free_group('x, y')
    f = FpGroup(F, [x ** 3, y ** 5, (x * y) ** 2])
    H = [x * y, x ** (-1) * y ** (-1) * x * y * x]
    p1 = reidemeister_presentation(f, H)
    assert str(p1) == '((y_1, y_2), (y_1**2, y_2**3, y_2*y_1*y_2*y_1*y_2*y_1))'
    H = f.subgroup(H)
    assert (H.generators, H.relators) == p1
    f = FpGroup(F, [x ** 3, y ** 3, (x * y) ** 3])
    H = [x * y, x * y ** (-1)]
    p2 = reidemeister_presentation(f, H)
    assert str(p2) == '((x_0, y_0), (x_0**3, y_0**3, x_0*y_0*x_0*y_0*x_0*y_0))'
    f = FpGroup(F, [x ** 2 * y ** 2, y ** (-1) * x * y * x ** (-3)])
    H = [x]
    p3 = reidemeister_presentation(f, H)
    assert str(p3) == '((x_0,), (x_0**4,))'
    f = FpGroup(F, [x ** 3 * y ** (-3), (x * y) ** 3, (x * y ** (-1)) ** 2])
    H = [x]
    p4 = reidemeister_presentation(f, H)
    assert str(p4) == '((x_0,), (x_0**6,))'
    F, a, b, c = free_group('a, b, c')
    f = FpGroup(F, [a ** 11, b ** 5, c ** 4, (b * c ** 2) ** 2, (a * b * c) ** 3, (a ** 4 * c ** 2) ** 3, b ** 2 * c ** (-1) * b ** (-1) * c, a ** 4 * b ** (-1) * a ** (-1) * b])
    H = [a, b, c ** 2]
    gens, rels = reidemeister_presentation(f, H)
    assert str(gens) == '(b_1, c_3)'
    assert len(rels) == 18