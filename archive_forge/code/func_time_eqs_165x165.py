from sympy.polys.rings import ring
from sympy.polys.fields import field
from sympy.polys.domains import ZZ, QQ
from sympy.polys.solvers import solve_lin_sys
def time_eqs_165x165():
    if len(eqs_165x165()) != 165:
        raise ValueError('length should be 165')