from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_nf_mora():
    f = sdm_from_dict({(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1), (1, 0, 2, 1): QQ(1), (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)}, grlex)
    f1 = sdm_from_dict({(1, 1, 1, 0): QQ(1), (1, 0, 2, 0): QQ(1), (1, 0, 0, 0): QQ(-1)}, grlex)
    f2 = sdm_from_dict({(1, 1, 1, 0): QQ(1)}, grlex)
    id0, id1, id2 = [sdm_from_dict({(i, 0, 0, 0): QQ(1)}, grlex) for i in range(3)]
    assert sdm_nf_mora(f, [f1, f2], grlex, QQ, phantom=(id0, [id1, id2])) == ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1)), ((1, 1, 0, 1), QQ(1))], [((1, 1, 0, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])
    assert sdm_nf_mora(f, [f2, f1], grlex, QQ, phantom=(id0, [id2, id1])) == ([((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))], [((2, 1, 0, 1), QQ(-1)), ((2, 0, 1, 1), QQ(-1)), ((0, 0, 0, 0), QQ(1))])
    f = sdm_from_vector([x * z, y ** 2 + y * z - z, y], lex, QQ, gens=[x, y, z])
    f1 = sdm_from_vector([x, y, 1], lex, QQ, gens=[x, y, z])
    f2 = sdm_from_vector([x * y, z, z ** 2], lex, QQ, gens=[x, y, z])
    assert sdm_nf_mora(f, [f1, f2], lex, QQ) == sdm_nf_mora(f, [f2, f1], lex, QQ) == [((1, 0, 1, 1), QQ(1)), ((1, 0, 0, 1), QQ(-1)), ((0, 1, 1, 0), QQ(-1)), ((0, 1, 0, 1), QQ(1))]