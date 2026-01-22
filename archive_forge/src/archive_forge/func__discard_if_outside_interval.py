from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _discard_if_outside_interval(f, M, inf, sup, K, negative, fast, mobius):
    """Discard an isolating interval if outside ``(inf, sup)``. """
    F = K.get_field()
    while True:
        u, v = _mobius_to_interval(M, F)
        if negative:
            u, v = (-v, -u)
        if (inf is None or u >= inf) and (sup is None or v <= sup):
            if not mobius:
                return (u, v)
            else:
                return (f, M)
        elif sup is not None and u > sup or (inf is not None and v < inf):
            return None
        else:
            f, M = dup_step_refine_real_root(f, M, K, fast=fast)