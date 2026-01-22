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
class piecewise_cc(TransformedPiecewiseLinearFunction):
    """Discrete CC piecewise representation

    Expresses a piecewise linear function using
    the CC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_cc, self).__init__(*args, **kwds)
        polytopes = range(len(self.breakpoints) - 1)
        vertices = range(len(self.breakpoints))

        def vertex_polys(v):
            if v == 0:
                return [v]
            if v == len(self.breakpoints) - 1:
                return [v - 1]
            else:
                return [v - 1, v]
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple((variable(lb=0) for v in vertices))
        y = self.v['y'] = variable_tuple((variable(domain_type=IntegerSet, lb=0, ub=1) for p in polytopes))
        lmbda_tuple = tuple(lmbda)
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=lmbda_tuple + (self.input,), coefficients=self.breakpoints + (-1,), rhs=0))
        self.c.append(linear_constraint(variables=lmbda_tuple + (self.output,), coefficients=self.values + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        self.c.append(linear_constraint(variables=lmbda_tuple, coefficients=(1,) * len(lmbda), rhs=1))
        clist = []
        for v in vertices:
            variables = tuple((y[p] for p in vertex_polys(v)))
            clist.append(linear_constraint(variables=variables + (lmbda[v],), coefficients=(1,) * len(variables) + (-1,), lb=0))
        self.c.append(constraint_tuple(clist))
        self.c.append(linear_constraint(variables=tuple(y), coefficients=(1,) * len(y), rhs=1))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_cc, self).validate(**kwds)