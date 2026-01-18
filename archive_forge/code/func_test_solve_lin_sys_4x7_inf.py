from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_solve_lin_sys_4x7_inf():
    domain, x1, x2, x3, x4, x5, x6, x7 = ring('x1,x2,x3,x4,x5,x6,x7', QQ)
    eqs = [x1 + 4 * x2 - x4 + 7 * x6 - 9 * x7 - 3, 2 * x1 + 8 * x2 - x3 + 3 * x4 + 9 * x5 - 13 * x6 + 7 * x7 - 9, 2 * x3 - 3 * x4 - 4 * x5 + 12 * x6 - 8 * x7 - 1, -x1 - 4 * x2 + 2 * x3 + 4 * x4 + 8 * x5 - 31 * x6 + 37 * x7 - 4]
    sol = {x1: 4 - 4 * x2 - 2 * x5 - x6 + 3 * x7, x3: 2 - x5 + 3 * x6 - 5 * x7, x4: 1 - 2 * x5 + 6 * x6 - 6 * x7}
    assert solve_lin_sys(eqs, domain) == sol