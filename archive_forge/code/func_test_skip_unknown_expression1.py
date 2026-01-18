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
def test_skip_unknown_expression1(self):
    if self.tightener is not fbbt:
        raise unittest.SkipTest('Appsi FBBT does not support unknown expressions yet')
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, 1))
    m.y = pyo.Var()
    expr = DummyExpr([m.x, m.y])
    m.c = pyo.Constraint(expr=expr == 1)
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo.contrib.fbbt.fbbt'):
        new_bounds = self.tightener(m)
    self.assertEqual(pyo.value(m.x.lb), 1)
    self.assertEqual(pyo.value(m.x.ub), 1)
    self.assertEqual(pyo.value(m.y.lb), None)
    self.assertEqual(pyo.value(m.y.ub), None)
    self.assertIn('Unsupported expression type for FBBT', OUT.getvalue())