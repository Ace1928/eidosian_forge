import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_ranged(self):
    m = self.make_model()
    visitor = ExpressionBoundsVisitor()
    self.assertEqual(visitor.walk_expression(inequality(m.z, m.y, 5)), (_true, _true))
    self.assertEqual(visitor.walk_expression(inequality(m.y, m.z, m.y)), (_false, _false))
    self.assertEqual(visitor.walk_expression(inequality(m.y, m.x, m.y)), (_false, _true))