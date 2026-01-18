from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_spoly():
    f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    g = [((2, 3, 0), QQ(1))]
    h = [((1, 2, 3), QQ(1))]
    assert sdm_spoly(f, h, lex, QQ) == []
    assert sdm_spoly(f, g, lex, QQ) == [((1, 2, 1), QQ(1))]