import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_unknown_classes(self):

    class UnknownNumeric(NumericExpression):
        pass

    class UnknownLogic(BooleanExpression):

        def nargs(self):
            return 0

    class UnknownOther(ExpressionBase):

        @property
        def args(self):
            return ()

        def nargs(self):
            return 0
    visitor = ExpressionBoundsVisitor()
    with LoggingIntercept() as LOG:
        self.assertEqual(visitor.walk_expression(UnknownNumeric(())), (-inf, inf))
    self.assertEqual(LOG.getvalue(), "Unexpected expression node type 'UnknownNumeric' found while walking expression tree; returning (-inf, inf) for the expression bounds.\n")
    with LoggingIntercept() as LOG:
        self.assertEqual(visitor.walk_expression(UnknownLogic(())), (_false, _true))
    self.assertEqual(LOG.getvalue(), "Unexpected expression node type 'UnknownLogic' found while walking expression tree; returning (False, True) for the expression bounds.\n")
    with self.assertRaisesRegex(DeveloperError, "Unexpected expression node type 'UnknownOther' found"):
        visitor.walk_expression(UnknownOther())