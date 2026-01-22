import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
@unittest.skipIf(not docplex_available, 'docplex is not available')
class TestCPExpressionWalker_NamedExpressions(CommonTest):

    def test_named_expression(self):
        m = self.get_model()
        m.e = Expression(expr=m.x ** 2 + 7)
        m.c = Constraint(expr=m.e <= 32)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(x ** 2 + 7))

    def test_repeated_named_expression(self):
        m = self.get_model()
        m.e = Expression(expr=m.x ** 2 + 7)
        m.c = Constraint(expr=m.e - 8 * m.e <= 32)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.x), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        self.assertTrue(expr[1].equals(x ** 2 + 7 + -1 * (8 * (x ** 2 + 7))))