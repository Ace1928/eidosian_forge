from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_solve_lin_sys_3x4_none():
    domain, x1, x2, x3, x4 = ring('x1,x2,x3,x4', QQ)
    eqs = [2 * x1 + x2 + 7 * x3 - 7 * x4 - 2, -3 * x1 + 4 * x2 - 5 * x3 - 6 * x4 - 3, x1 + x2 + 4 * x3 - 5 * x4 - 2]
    assert solve_lin_sys(eqs, domain) is None