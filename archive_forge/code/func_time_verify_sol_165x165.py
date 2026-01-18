from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_verify_sol_165x165():
    eqs = eqs_165x165()
    sol = sol_165x165()
    zeros = [eq.compose(sol) for eq in eqs]
    if not all((zero == 0 for zero in zeros)):
        raise ValueError('All should be 0')