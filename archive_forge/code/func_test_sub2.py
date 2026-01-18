import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.fileutils import find_library
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import (
import math
import platform
from io import StringIO
def test_sub2(self):
    if not numpy_available:
        raise unittest.SkipTest('Numpy is not available')
    x_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
    c_bounds = [(-2.5, 2.8), (-2.5, -0.5), (0.5, 2.8), (-2.5, 0), (0, 2.8), (-2.5, -1), (1, 2.8), (-1, -0.5), (0.5, 1)]
    for xl, xu in x_bounds:
        for cl, cu in c_bounds:
            m = pyo.Block(concrete=True)
            m.x = pyo.Var(bounds=(xl, xu))
            m.y = pyo.Var()
            m.c = pyo.Constraint(expr=pyo.inequality(body=m.y - m.x, lower=cl, upper=cu))
            self.tightener(m)
            x = np.linspace(pyo.value(m.x.lb), pyo.value(m.x.ub), 100)
            z = np.linspace(pyo.value(m.c.lower), pyo.value(m.c.upper), 100)
            if m.y.lb is None:
                yl = -np.inf
            else:
                yl = m.y.lb
            if m.y.ub is None:
                yu = np.inf
            else:
                yu = m.y.ub
            for _x in x:
                _y = z + _x
                self.assertTrue(np.all(yl <= _y))
                self.assertTrue(np.all(yu >= _y))