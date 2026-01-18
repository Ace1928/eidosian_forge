from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_to_expr_eqs_10x8():
    eqs = eqs_10x8()
    assert [R_8.from_expr(eq.as_expr()) for eq in eqs] == eqs