from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_deg():
    assert sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)]) == 7