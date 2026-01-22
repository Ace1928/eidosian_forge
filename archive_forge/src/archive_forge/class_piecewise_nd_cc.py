from collections.abc import Sized
import logging
from pyomo.core.kernel.block import block
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.variable import variable, variable_dict, variable_tuple
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.expression import expression, expression_tuple
import pyomo.core.kernel.piecewise_library.util
class piecewise_nd_cc(TransformedPiecewiseLinearFunctionND):
    """Discrete CC multi-variate piecewise representation

    Expresses a multi-variate piecewise linear function
    using the CC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_nd_cc, self).__init__(*args, **kwds)
        ndim = len(self.input)
        nsimplices = len(self.triangulation.simplices)
        npoints = len(self.triangulation.points)
        pointsT = list(zip(*self.triangulation.points))
        dimensions = range(ndim)
        simplices = range(nsimplices)
        vertices = range(npoints)
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple((variable(lb=0) for v in vertices))
        y = self.v['y'] = variable_tuple((variable(domain_type=IntegerSet, lb=0, ub=1) for s in simplices))
        lmbda_tuple = tuple(lmbda)
        self.c = constraint_list()
        clist = []
        for d in dimensions:
            clist.append(linear_constraint(variables=lmbda_tuple + (self.input[d],), coefficients=tuple(pointsT[d]) + (-1,), rhs=0))
        self.c.append(constraint_tuple(clist))
        del clist
        self.c.append(linear_constraint(variables=lmbda_tuple + (self.output,), coefficients=tuple(self.values) + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        self.c.append(linear_constraint(variables=lmbda_tuple, coefficients=(1,) * len(lmbda_tuple), rhs=1))
        vertex_to_simplex = [[] for v in vertices]
        for s, simplex in enumerate(self.triangulation.simplices):
            for v in simplex:
                vertex_to_simplex[v].append(s)
        clist = []
        for v in vertices:
            variables = tuple((y[s] for s in vertex_to_simplex[v]))
            clist.append(linear_constraint(variables=variables + (lmbda[v],), coefficients=(1,) * len(variables) + (-1,), lb=0))
        self.c.append(constraint_tuple(clist))
        del clist
        self.c.append(linear_constraint(variables=y, coefficients=(1,) * len(y), rhs=1))