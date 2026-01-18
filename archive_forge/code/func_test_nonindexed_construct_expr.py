import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_nonindexed_construct_expr(self):
    model = ConcreteModel()
    model.e = Expression(expr=Expression.Skip)
    self.assertEqual(len(model.e), 0)
    model.del_component(model.e)
    model.e = Expression()
    self.assertEqual(model.e.extract_values(), {None: None})
    model.del_component(model.e)
    model.e = Expression(expr=1.0)
    self.assertEqual(model.e.extract_values(), {None: 1.0})
    model.del_component(model.e)
    model.e = Expression(expr={None: 1.0})
    self.assertEqual(model.e.extract_values(), {None: 1.0})
    with self.assertRaises(KeyError):
        model.e.add(2, 2)