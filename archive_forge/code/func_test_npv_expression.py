import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_npv_expression(self):
    m = self.make_model()
    m.p = Param(initialize=4, mutable=True)
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(1 / m.p)
    self.assertEqual(lb, 0.25)
    self.assertEqual(ub, 0.25)