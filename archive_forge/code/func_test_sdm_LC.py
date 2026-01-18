from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_LC():
    assert sdm_LC([((1, 2, 3), QQ(5))], QQ) == QQ(5)