from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
def _body_function_variables(self, values=False):
    """Returns variables in the order they should be
        passed to the body function. If values is True, then
        return the current value of each variable in place
        of the variables themselves."""
    if not values:
        return (self.r1, self.r2, self.x)
    else:
        return (self.r1.value, self.r2.value, tuple((xi.value for xi in self.x)))