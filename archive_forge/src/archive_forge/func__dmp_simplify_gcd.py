from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
def _dmp_simplify_gcd(f, g, u, K):
    """Try to eliminate `x_0` from GCD computation in `K[X]`. """
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)
    if df > 0 and dg > 0:
        return None
    if not (df or dg):
        F = dmp_LC(f, K)
        G = dmp_LC(g, K)
    elif not df:
        F = dmp_LC(f, K)
        G = dmp_content(g, u, K)
    else:
        F = dmp_content(f, u, K)
        G = dmp_LC(g, K)
    v = u - 1
    h = dmp_gcd(F, G, v, K)
    cff = [dmp_quo(cf, h, v, K) for cf in f]
    cfg = [dmp_quo(cg, h, v, K) for cg in g]
    return ([h], cff, cfg)