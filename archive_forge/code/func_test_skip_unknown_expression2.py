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
def test_skip_unknown_expression2(self):
    if self.tightener is not fbbt:
        raise unittest.SkipTest('Appsi FBBT does not support unknown expressions yet')

    def dummy_unary_expr(x):
        return 0.5 * x
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 4))
    expr = UnaryFunctionExpression((m.x,), name='dummy_unary_expr', fcn=dummy_unary_expr)
    m.c = pyo.Constraint(expr=expr == 1)
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo.contrib.fbbt.fbbt'):
        new_bounds = self.tightener(m)
    self.assertEqual(pyo.value(m.x.lb), 0)
    self.assertEqual(pyo.value(m.x.ub), 4)
    self.assertIn('Unsupported expression type for FBBT', OUT.getvalue())