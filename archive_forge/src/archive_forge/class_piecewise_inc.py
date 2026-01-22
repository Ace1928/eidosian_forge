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
class piecewise_inc(TransformedPiecewiseLinearFunction):
    """Discrete INC piecewise representation

    Expresses a piecewise linear function using
    the INC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_inc, self).__init__(*args, **kwds)
        polytopes = range(len(self.breakpoints) - 1)
        self.v = variable_dict()
        delta = self.v['delta'] = variable_tuple((variable() for p in polytopes))
        delta[0].ub = 1
        delta[-1].lb = 0
        delta_tuple = tuple(delta)
        y = self.v['y'] = variable_tuple((variable(domain_type=IntegerSet, lb=0, ub=1) for p in polytopes[:-1]))
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=(self.input,) + delta_tuple, coefficients=(-1,) + tuple((self.breakpoints[p + 1] - self.breakpoints[p] for p in polytopes)), rhs=-self.breakpoints[0]))
        self.c.append(linear_constraint(variables=(self.output,) + delta_tuple, coefficients=(-1,) + tuple((self.values[p + 1] - self.values[p] for p in polytopes))))
        if self.bound == 'ub':
            self.c[-1].lb = -self.values[0]
        elif self.bound == 'lb':
            self.c[-1].ub = -self.values[0]
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = -self.values[0]
        clist1 = []
        clist2 = []
        for p in polytopes[:-1]:
            clist1.append(linear_constraint(variables=(delta[p + 1], y[p]), coefficients=(1, -1), ub=0))
            clist2.append(linear_constraint(variables=(y[p], delta[p]), coefficients=(1, -1), ub=0))
        self.c.append(constraint_tuple(clist1))
        self.c.append(constraint_tuple(clist2))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_inc, self).validate(**kwds)