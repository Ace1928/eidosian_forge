from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_res_q():
    x = var('x')
    assert res_q(3, 5, x) == 1