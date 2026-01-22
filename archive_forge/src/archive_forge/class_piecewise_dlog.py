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
class piecewise_dlog(TransformedPiecewiseLinearFunction):
    """Discrete DLOG piecewise representation

    Expresses a piecewise linear function using the DLOG
    formulation. This formulation uses logarithmic number of
    discrete variables in terms of number of breakpoints.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_dlog, self).__init__(*args, **kwds)
        breakpoints = self.breakpoints
        values = self.values
        if not is_positive_power_of_two(len(breakpoints) - 1):
            raise ValueError('The list of breakpoints must be of length (2^n)+1 for some positive integer n. Invalid length: %s' % len(breakpoints))
        L = log2floor(len(breakpoints) - 1)
        assert 2 ** L == len(breakpoints) - 1
        B_LEFT, B_RIGHT = self._branching_scheme(L)
        polytopes = range(len(breakpoints) - 1)
        vertices = range(len(breakpoints))

        def polytope_verts(p):
            return range(p, p + 2)
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_dict((((p, v), variable(lb=0)) for p in polytopes for v in polytope_verts(p)))
        y = self.v['y'] = variable_tuple((variable(domain_type=IntegerSet, lb=0, ub=1) for i in range(L)))
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=(self.input,) + tuple((lmbda[p, v] for p in polytopes for v in polytope_verts(p))), coefficients=(-1,) + tuple((breakpoints[v] for p in polytopes for v in polytope_verts(p))), rhs=0))
        self.c.append(linear_constraint(variables=(self.output,) + tuple((lmbda[p, v] for p in polytopes for v in polytope_verts(p))), coefficients=(-1,) + tuple((values[v] for p in polytopes for v in polytope_verts(p)))))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        self.c.append(linear_constraint(variables=tuple(lmbda.values()), coefficients=(1,) * len(lmbda), rhs=1))
        clist = []
        for i in range(L):
            variables = tuple((lmbda[p, v] for p in B_LEFT[i] for v in polytope_verts(p)))
            clist.append(linear_constraint(variables=variables + (y[i],), coefficients=(1,) * len(variables) + (-1,), ub=0))
        self.c.append(constraint_tuple(clist))
        del clist
        clist = []
        for i in range(L):
            variables = tuple((lmbda[p, v] for p in B_RIGHT[i] for v in polytope_verts(p)))
            clist.append(linear_constraint(variables=variables + (y[i],), coefficients=(1,) * len(variables) + (1,), ub=1))
        self.c.append(constraint_tuple(clist))

    def _branching_scheme(self, L):
        N = 2 ** L
        B_LEFT = []
        for i in range(1, L + 1):
            start = 1
            step = N // 2 ** i
            tmp = []
            while start < N:
                tmp.extend((j - 1 for j in range(start, start + step)))
                start += 2 * step
            B_LEFT.append(tmp)
        biglist = range(N)
        B_RIGHT = []
        for i in range(len(B_LEFT)):
            tmp = []
            for j in biglist:
                if j not in B_LEFT[i]:
                    tmp.append(j)
            B_RIGHT.append(sorted(tmp))
        return (B_LEFT, B_RIGHT)

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_dlog, self).validate(**kwds)