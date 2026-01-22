from sympy.core.random import _randint
from sympy.polys.galoistools import (
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.polyutils import _sort_factors
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
from sympy.utilities import subsets
from math import ceil as _ceil, log as _log
def _dup_cyclotomic_decompose(n, K):
    from sympy.ntheory import factorint
    H = [[K.one, -K.one]]
    for p, k in factorint(n).items():
        Q = [dup_quo(dup_inflate(h, p, K), h, K) for h in H]
        H.extend(Q)
        for i in range(1, k):
            Q = [dup_inflate(q, p, K) for q in Q]
            H.extend(Q)
    return H