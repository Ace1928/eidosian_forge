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
class TestCPExpressionWalker_PrecedenceExpressions(CommonTest):

    def test_start_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[1].start_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.start_before_start(i, i21, 0)))

    def test_start_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[1].end_time, delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.start_before_end(i, i21, 3)))

    def test_end_before_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].start_time, delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.end_before_start(i, i21, -2)))

    def test_end_before_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].end_time, delay=6))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.end_before_end(i, i21, 6)))

    def test_start_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[1].start_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.start_at_start(i, i21, 0)))

    def test_start_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[1].end_time, delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.start_at_end(i, i21, 3)))

    def test_end_at_start(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.at(m.i2[1].start_time, delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.end_at_start(i, i21, -2)))

    def test_end_at_end(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.i.end_time.at(m.i2[1].end_time, delay=6))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        self.assertIn(id(m.i), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        i = visitor.var_map[id(m.i)]
        i21 = visitor.var_map[id(m.i2[1])]
        self.assertTrue(expr[1].equals(cp.end_at_end(i, i21, 6)))

    def test_indirection_before_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.before(m.i.end_time, delay=3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.element([cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1) + 3 <= cp.end_of(i)))

    def test_indirection_after_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.after(m.i.end_time, delay=-2))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.end_of(i) + -2 <= cp.element([cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1)))

    def test_indirection_at_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].start_time.at(m.i.end_time, delay=4))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.element([cp.start_of(i21), cp.start_of(i22)], 0 + 1 * (y - 1) // 1) == cp.end_of(i) + 4))

    def test_before_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i.start_time.before(m.i2[m.y].end_time, delay=-4))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.start_of(i) + -4 <= cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)))

    def test_after_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i.start_time.after(m.i2[m.y].end_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1) + 0 <= cp.start_of(i)))

    def test_at_indirection_constraint(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i.start_time.at(m.i2[m.y].end_time, delay=-6))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.start_of(i) + -6 == cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)))

    def test_double_indirection_before_constraint(self):
        m = self.get_model()
        m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i3[1, m.x - 3].start_time.before(m.i2[m.y].end_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1, 3]), visitor.var_map)
        self.assertIn(id(m.i3[1, 4]), visitor.var_map)
        self.assertIn(id(m.i3[1, 5]), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1, 3])]
        i34 = visitor.var_map[id(m.i3[1, 4])]
        i35 = visitor.var_map[id(m.i3[1, 5])]
        self.assertTrue(expr[1].equals(cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)], 0 + 1 * (x + -3 - 3) // 1) <= cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)))

    def test_double_indirection_after_constraint(self):
        m = self.get_model()
        m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i3[1, m.x - 3].start_time.after(m.i2[m.y].end_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1, 3]), visitor.var_map)
        self.assertIn(id(m.i3[1, 4]), visitor.var_map)
        self.assertIn(id(m.i3[1, 5]), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1, 3])]
        i34 = visitor.var_map[id(m.i3[1, 4])]
        i35 = visitor.var_map[id(m.i3[1, 5])]
        self.assertTrue(expr[1].equals(cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1) <= cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)], 0 + 1 * (x + -3 - 3) // 1)))

    def test_double_indirection_at_constraint(self):
        m = self.get_model()
        m.i3 = IntervalVar([(1, 3), (1, 4), (1, 5)], length=4, optional=True)
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i3[1, m.x - 3].start_time.at(m.i2[m.y].end_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.i3[1, 3]), visitor.var_map)
        self.assertIn(id(m.i3[1, 4]), visitor.var_map)
        self.assertIn(id(m.i3[1, 5]), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        x = visitor.var_map[id(m.x)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        i33 = visitor.var_map[id(m.i3[1, 3])]
        i34 = visitor.var_map[id(m.i3[1, 4])]
        i35 = visitor.var_map[id(m.i3[1, 5])]
        self.assertTrue(expr[1].equals(cp.element([cp.start_of(i33), cp.start_of(i34), cp.start_of(i35)], 0 + 1 * (x + -3 - 3) // 1) == cp.element([cp.end_of(i21), cp.end_of(i22)], 0 + 1 * (y - 1) // 1)))

    def test_indirection_nonconstant_step_size(self):
        m = ConcreteModel()

        def param_rule(m, i):
            return i + 1
        m.p = Param([1, 3, 4], initialize=param_rule)
        m.x = Var(within={1, 3, 4})
        e = m.p[m.x]
        visitor = self.get_visitor()
        with self.assertRaisesRegex(ValueError, "Variable indirection 'p\\[x\\]' is over a discrete domain without a constant step size. This is not supported."):
            expr = visitor.walk_expression((e, e, 0))

    def test_indirection_with_param(self):
        m = ConcreteModel()

        def param_rule(m, i):
            return i + 1
        m.p = Param([1, 3, 5], initialize=param_rule)
        m.x = Var(within={1, 3, 5})
        m.a = Var(domain=Integers, bounds=(0, 100))
        e = m.p[m.x] / m.a
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.a), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        a = visitor.var_map[id(m.a)]
        self.assertTrue(expr[1].equals(cp.element([2, 4, 6], 0 + 1 * (x - 1) // 2) / a))