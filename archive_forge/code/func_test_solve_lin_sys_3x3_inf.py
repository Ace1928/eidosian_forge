from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_solve_lin_sys_3x3_inf():
    domain, x1, x2, x3 = ring('x1,x2,x3', QQ)
    eqs = [x1 - x2 + 2 * x3 - 1, 2 * x1 + x2 + x3 - 8, x1 + x2 - 5]
    sol = {x1: -x3 + 3, x2: x3 + 2}
    assert solve_lin_sys(eqs, domain) == sol