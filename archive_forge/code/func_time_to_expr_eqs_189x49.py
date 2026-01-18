from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_to_expr_eqs_189x49():
    eqs = eqs_189x49()
    assert [R_49.from_expr(eq.as_expr()) for eq in eqs] == eqs