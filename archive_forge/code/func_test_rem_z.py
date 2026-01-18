from sympy.core.symbol import var
from sympy.polys.polytools import (pquo, prem, sturm, subresultants)
from sympy.matrices import Matrix
from sympy.polys.subresultants_qq_zz import (sylvester, res, res_q, res_z, bezout,
def test_rem_z():
    x = var('x')
    p = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    q = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    assert rem_z(p, -q, x) != prem(p, -q, x)