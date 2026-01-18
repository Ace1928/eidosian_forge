import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_sums_of_squares_bounds(self):
    m = ConcreteModel()
    m.x = Var([1, 2], bounds=(-2, 6))
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(m.x[1] * m.x[1] + m.x[2] * m.x[2])
    self.assertEqual(lb, 0)
    self.assertEqual(ub, 72)