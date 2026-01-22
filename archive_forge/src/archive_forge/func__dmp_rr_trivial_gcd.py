from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
def _dmp_rr_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)
    if_contain_one = dmp_one_p(f, u, K) or dmp_one_p(g, u, K)
    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        if K.is_nonnegative(dmp_ground_LC(g, u, K)):
            return (g, dmp_zero(u), dmp_one(u, K))
        else:
            return (dmp_neg(g, u, K), dmp_zero(u), dmp_ground(-K.one, u))
    elif zero_g:
        if K.is_nonnegative(dmp_ground_LC(f, u, K)):
            return (f, dmp_one(u, K), dmp_zero(u))
        else:
            return (dmp_neg(f, u, K), dmp_ground(-K.one, u), dmp_zero(u))
    elif if_contain_one:
        return (dmp_one(u, K), f, g)
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None