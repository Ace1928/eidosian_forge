import logging
import bisect
from pyomo.core.expr.numvalue import value as _value
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.block import block
from pyomo.core.kernel.expression import expression, expression_tuple
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.sos import sos2
from pyomo.core.kernel.piecewise_library.util import (
def _branching_scheme(self, n):
    N = 2 ** n
    S = range(n)
    G = generate_gray_code(n)
    L = tuple(([k for k in range(N + 1) if (k == 0 or G[k - 1][s] == 1) and (k == N or G[k][s] == 1)] for s in S))
    R = tuple(([k for k in range(N + 1) if (k == 0 or G[k - 1][s] == 0) and (k == N or G[k][s] == 0)] for s in S))
    return (S, L, R)