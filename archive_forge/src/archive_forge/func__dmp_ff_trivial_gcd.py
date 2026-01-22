from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
def _dmp_ff_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a field. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)
    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        return (dmp_ground_monic(g, u, K), dmp_zero(u), dmp_ground(dmp_ground_LC(g, u, K), u))
    elif zero_g:
        return (dmp_ground_monic(f, u, K), dmp_ground(dmp_ground_LC(f, u, K), u), dmp_zero(u))
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None