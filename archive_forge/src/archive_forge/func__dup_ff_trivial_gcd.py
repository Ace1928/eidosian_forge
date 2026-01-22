from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
def _dup_ff_trivial_gcd(f, g, K):
    """Handle trivial cases in GCD algorithm over a field. """
    if not (f or g):
        return ([], [], [])
    elif not f:
        return (dup_monic(g, K), [], [dup_LC(g, K)])
    elif not g:
        return (dup_monic(f, K), [dup_LC(f, K)], [])
    else:
        return None