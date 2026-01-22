from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
def _dmp_zz_gcd_interpolate(h, x, v, K):
    """Interpolate polynomial GCD from integer GCD. """
    f = []
    while not dmp_zero_p(h, v):
        g = dmp_ground_trunc(h, x, v, K)
        f.insert(0, g)
        h = dmp_sub(h, g, v, K)
        h = dmp_quo_ground(h, x, v, K)
    if K.is_negative(dmp_ground_LC(f, v + 1, K)):
        return dmp_neg(f, v + 1, K)
    else:
        return f