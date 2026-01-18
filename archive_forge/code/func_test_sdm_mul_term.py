from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_mul_term():
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ) == []
    assert sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ) == []
    assert sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ) == [((1, 1, 0), QQ(1))]
    f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    assert sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ) == [((2, 1, 2), QQ(8)), ((1, 2, 1), QQ(6))]