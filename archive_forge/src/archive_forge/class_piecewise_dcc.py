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
class piecewise_dcc(TransformedPiecewiseLinearFunction):
    """Discrete DCC piecewise representation

    Expresses a piecewise linear function using
    the DCC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_dcc, self).__init__(*args, **kwds)
        polytopes = range(len(self.breakpoints) - 1)
        vertices = range(len(self.breakpoints))

        def polytope_verts(p):
            return range(p, p + 2)
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_dict((((p, v), variable(lb=0)) for p in polytopes for v in vertices))
        y = self.v['y'] = variable_tuple((variable(domain_type=IntegerSet, lb=0, ub=1) for p in polytopes))
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=tuple((lmbda[p, v] for p in polytopes for v in polytope_verts(p))) + (self.input,), coefficients=tuple((self.breakpoints[v] for p in polytopes for v in polytope_verts(p))) + (-1,), rhs=0))
        self.c.append(linear_constraint(variables=tuple((lmbda[p, v] for p in polytopes for v in polytope_verts(p))) + (self.output,), coefficients=tuple((self.values[v] for p in polytopes for v in polytope_verts(p))) + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        clist = []
        for p in polytopes:
            variables = tuple((lmbda[p, v] for v in polytope_verts(p)))
            clist.append(linear_constraint(variables=variables + (y[p],), coefficients=(1,) * len(variables) + (-1,), rhs=0))
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
        return super(piecewise_dcc, self).validate(**kwds)