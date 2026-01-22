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
class TestCPExpressionWalker_IntervalVars(CommonTest):

    def test_interval_var_fixed_presences_correct(self):
        m = self.get_model()
        m.silly = LogicalConstraint(expr=m.i.is_present)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.silly.expr, m.silly, 0))
        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        self.assertTrue(i.is_optional())
        m.i.is_present.fix(False)
        m.c = LogicalConstraint(expr=m.i.is_present.lor(m.i2[1].start_time == 2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        self.assertTrue(i.is_absent())
        self.assertTrue(i21.is_present())

    def test_interval_var_fixed_length(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(2, 7), end=(6, 11), optional=True)
        m.i.length.fix(4)
        m.silly = LogicalConstraint(expr=m.i.is_present)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.silly.expr, m.silly, 0))
        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        self.assertTrue(i.is_optional())
        self.assertEqual(i.get_length(), (4, 4))
        self.assertEqual(i.get_start(), (2, 7))
        self.assertEqual(i.get_end(), (6, 11))

    def test_interval_var_fixed_start_and_end(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(3, 7), end=(6, 10))
        m.i.start_time.fix(3)
        m.i.end_time.fix(6)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.i, m.i, 0))
        self.assertIn(id(m.i), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        self.assertFalse(i.is_optional())
        self.assertEqual(i.get_start(), (3, 3))
        self.assertEqual(i.get_end(), (6, 6))