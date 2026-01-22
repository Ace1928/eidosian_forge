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
class piecewise_sos2(TransformedPiecewiseLinearFunction):
    """Discrete SOS2 piecewise representation

    Expresses a piecewise linear function using
    the SOS2 formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_sos2, self).__init__(*args, **kwds)
        y_tuple = tuple((variable(lb=0) for i in range(len(self.breakpoints))))
        y = self.v = variable_tuple(y_tuple)
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=y_tuple + (self.input,), coefficients=self.breakpoints + (-1,), rhs=0))
        self.c.append(linear_constraint(variables=y_tuple + (self.output,), coefficients=self.values + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        self.c.append(linear_constraint(variables=y_tuple, coefficients=(1,) * len(y), rhs=1))
        self.s = sos2(y)

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_sos2, self).validate(**kwds)