from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_sdm_monomial_lcm():
    assert sdm_monomial_lcm((1, 2, 3), (1, 5, 0)) == (1, 5, 3)