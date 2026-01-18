from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_from_dict():
    dic = {(1, 2, 1, 1): QQ(1), (1, 1, 2, 1): QQ(1), (1, 0, 2, 1): QQ(1), (1, 0, 0, 3): QQ(1), (1, 1, 1, 0): QQ(1)}
    assert sdm_from_dict(dic, grlex) == [((1, 2, 1, 1), QQ(1)), ((1, 1, 2, 1), QQ(1)), ((1, 0, 2, 1), QQ(1)), ((1, 0, 0, 3), QQ(1)), ((1, 1, 1, 0), QQ(1))]