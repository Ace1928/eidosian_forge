from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_ecart():
    assert sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)]) == 0
    assert sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)]) == 3