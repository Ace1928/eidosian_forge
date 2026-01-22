from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _disjoint_p(M, N, strict=False):
    """Check if Mobius transforms define disjoint intervals. """
    a1, b1, c1, d1 = M
    a2, b2, c2, d2 = N
    a1d1, b1c1 = (a1 * d1, b1 * c1)
    a2d2, b2c2 = (a2 * d2, b2 * c2)
    if a1d1 == b1c1 and a2d2 == b2c2:
        return True
    if a1d1 > b1c1:
        a1, c1, b1, d1 = (b1, d1, a1, c1)
    if a2d2 > b2c2:
        a2, c2, b2, d2 = (b2, d2, a2, c2)
    if not strict:
        return a2 * d1 >= c2 * b1 or b2 * c1 <= d2 * a1
    else:
        return a2 * d1 > c2 * b1 or b2 * c1 < d2 * a1