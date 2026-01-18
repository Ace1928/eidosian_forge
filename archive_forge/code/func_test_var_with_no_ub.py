import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_var_with_no_ub(self):
    m = self.make_model()
    m.y.setub(None)
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(m.x - m.y)
    self.assertEqual(lb, -float('inf'))
    self.assertEqual(ub, 1)