from pyomo.common.dependencies import (
from pyomo.core.expr.numvalue import NumericValue, value
from pyomo.core.kernel.constraint import IConstraint, constraint_tuple
min(lslack, uslack)