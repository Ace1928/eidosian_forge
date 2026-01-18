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
def vertex_polys(v):
    if v == 0:
        return [v]
    if v == len(self.breakpoints) - 1:
        return [v - 1]
    else:
        return [v - 1, v]