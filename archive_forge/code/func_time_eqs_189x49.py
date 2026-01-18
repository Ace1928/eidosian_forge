from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_eqs_189x49():
    if len(eqs_189x49()) != 189:
        raise ValueError('Length should be equal to 189')