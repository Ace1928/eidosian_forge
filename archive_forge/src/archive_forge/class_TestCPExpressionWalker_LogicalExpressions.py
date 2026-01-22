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
class TestCPExpressionWalker_LogicalExpressions(CommonTest):

    def test_write_logical_and(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b.land(m.b2['b']))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.b2['b']), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        b2b = visitor.var_map[id(m.b2['b'])]
        self.assertTrue(expr[1].equals(cp.logical_and(b, b2b)))

    def test_write_logical_or(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b.lor(m.i.is_present))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.logical_or(b, cp.presence_of(i))))

    def test_write_xor(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b.xor(m.i2[2].start_time >= 5))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i22 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cp.count([b, cp.less_or_equal(5, cp.start_of(i22))], 1) == 1))

    def test_write_logical_not(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=~m.b2['a'])
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b2['a']), visitor.var_map)
        b2a = visitor.var_map[id(m.b2['a'])]
        self.assertTrue(expr[1].equals(cp.logical_not(b2a)))

    def test_equivalence(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=equivalent(~m.b2['a'], m.b))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.b2['a']), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        b2a = visitor.var_map[id(m.b2['a'])]
        self.assertTrue(expr[1].equals(cp.equal(cp.logical_not(b2a), b)))

    def test_implication(self):
        m = self.get_model()
        m.c = LogicalConstraint(expr=m.b2['a'].implies(~m.b))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.b2['a']), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        b2a = visitor.var_map[id(m.b2['a'])]
        self.assertTrue(expr[1].equals(cp.if_then(b2a, cp.logical_not(b))))

    def test_equality(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.b.implies(m.a[3] == 4))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]
        self.assertTrue(expr[1].equals(cp.if_then(b, cp.equal(a3, 4))))

    def test_inequality(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.b.implies(m.a[3] >= m.a[4]))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        self.assertIn(id(m.a[4]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]
        a4 = visitor.var_map[id(m.a[4])]
        self.assertTrue(expr[1].equals(cp.if_then(b, cp.less_or_equal(a4, a3))))

    def test_ranged_inequality(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = Constraint(expr=inequality(3, m.a[2], 5))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.a[2]), visitor.var_map)
        a2 = visitor.var_map[id(m.a[2])]
        self.assertTrue(expr[1].equals(cp.range(a2, 3, 5)))

    def test_not_equal(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.b.implies(NotEqualExpression([m.a[3], m.a[4]])))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.a[3]), visitor.var_map)
        self.assertIn(id(m.a[4]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        a3 = visitor.var_map[id(m.a[3])]
        a4 = visitor.var_map[id(m.a[4])]
        self.assertTrue(expr[1].equals(cp.if_then(b, a3 != a4)))

    def test_exactly_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=exactly(3, [m.a[i] == 4 for i in m.I]))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.equal(cp.count([a[i] == 4 for i in m.I], 1), 3)))

    def test_atleast_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=atleast(3, [m.a[i] == 4 for i in m.I]))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.greater_or_equal(cp.count([a[i] == 4 for i in m.I], 1), 3)))

    def test_atmost_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=atmost(3, [m.a[i] == 4 for i in m.I]))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.less_or_equal(cp.count([a[i] == 4 for i in m.I], 1), 3)))

    def test_all_diff_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.a.bounds = (11, 20)
        m.c = LogicalConstraint(expr=all_different(m.a))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.all_diff((a[i] for i in m.I))))

    def test_count_if_expression(self):
        m = self.get_model()
        m.a.domain = Integers
        m.a.bounds = (11, 20)
        m.c = Constraint(expr=count_if((m.a[i] == i for i in m.I)) == 5)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        a = {}
        for i in m.I:
            self.assertIn(id(m.a[i]), visitor.var_map)
            a[i] = visitor.var_map[id(m.a[i])]
        self.assertTrue(expr[1].equals(cp.count((a[i] == i for i in m.I), 1) == 5))

    def test_interval_var_is_present(self):
        m = self.get_model()
        m.a.domain = Integers
        m.c = LogicalConstraint(expr=m.i.is_present.implies(m.a[1] == 5))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.a[1]), visitor.var_map)
        self.assertIn(id(m.i), visitor.var_map)
        a1 = visitor.var_map[id(m.a[1])]
        i = visitor.var_map[id(m.i)]
        self.assertTrue(expr[1].equals(cp.if_then(cp.presence_of(i), a1 == 5)))

    def test_interval_var_is_present_indirection(self):
        m = self.get_model()
        m.a.domain = Integers
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].is_present.implies(m.a[1] >= 7))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.a[1]), visitor.var_map)
        a1 = visitor.var_map[id(m.a[1])]
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cp.if_then(cp.element([cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1) == True, cp.less_or_equal(7, a1))))

    def test_is_present_indirection_and_length(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=[1, 2])
        m.c = LogicalConstraint(expr=m.i2[m.y].is_present.land(m.i2[m.y].length >= 7))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.expr, m.c, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cp.logical_and(cp.element([cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1) == True, cp.less_or_equal(7, cp.element([cp.length_of(i21), cp.length_of(i22)], 0 + 1 * (y - 1) // 1)))))

    def test_handle_getattr_lor(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=(1, 2))
        e = m.i2[m.y].is_present.lor(~m.b)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.b), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        b = visitor.var_map[id(m.b)]
        self.assertTrue(expr[1].equals(cp.logical_or(cp.element([cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1) == True, cp.logical_not(b))))

    def test_handle_getattr_xor(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=(1, 2))
        e = m.i2[m.y].is_present.xor(m.b)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.b), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        b = visitor.var_map[id(m.b)]
        self.assertTrue(expr[1].equals(cp.equal(cp.count([cp.element([cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1) == True, b], 1), 1)))

    def test_handle_getattr_equivalent_to(self):
        m = self.get_model()
        m.y = Var(domain=Integers, bounds=(1, 2))
        e = m.i2[m.y].is_present.equivalent_to(~m.b)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.y), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        self.assertIn(id(m.b), visitor.var_map)
        y = visitor.var_map[id(m.y)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        b = visitor.var_map[id(m.b)]
        self.assertTrue(expr[1].equals(cp.equal(cp.element([cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1) == True, cp.logical_not(b))))

    def test_logical_or_on_indirection(self):
        m = ConcreteModel()
        m.b = BooleanVar([2, 3, 4, 5])
        m.x = Var(domain=Integers, bounds=(3, 5))
        e = m.b[m.x].lor(m.x == 5)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.b[3]), visitor.var_map)
        self.assertIn(id(m.b[4]), visitor.var_map)
        self.assertIn(id(m.b[5]), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        b3 = visitor.var_map[id(m.b[3])]
        b4 = visitor.var_map[id(m.b[4])]
        b5 = visitor.var_map[id(m.b[5])]
        self.assertTrue(expr[1].equals(cp.logical_or(cp.element([b3, b4, b5], 0 + 1 * (x - 3) // 1) == True, cp.equal(x, 5))))

    def test_logical_xor_on_indirection(self):
        m = ConcreteModel()
        m.b = BooleanVar([2, 3, 4, 5])
        m.b[4].fix(False)
        m.x = Var(domain=Integers, bounds=(3, 5))
        e = m.b[m.x].xor(m.x == 5)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.x), visitor.var_map)
        self.assertIn(id(m.b[3]), visitor.var_map)
        self.assertIn(id(m.b[5]), visitor.var_map)
        x = visitor.var_map[id(m.x)]
        b3 = visitor.var_map[id(m.b[3])]
        b5 = visitor.var_map[id(m.b[5])]
        self.assertTrue(expr[1].equals(cp.equal(cp.count([cp.element([b3, False, b5], 0 + 1 * (x - 3) // 1) == True, cp.equal(x, 5)], 1), 1)))

    def test_using_precedence_expr_as_boolean_expr(self):
        m = self.get_model()
        e = m.b.implies(m.i2[2].start_time.before(m.i2[1].start_time))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cp.if_then(b, cp.start_of(i22) + 0 <= cp.start_of(i21))))

    def test_using_precedence_expr_as_boolean_expr_positive_delay(self):
        m = self.get_model()
        e = m.b.implies(m.i2[2].start_time.before(m.i2[1].start_time, delay=4))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cp.if_then(b, cp.start_of(i22) + 4 <= cp.start_of(i21))))

    def test_using_precedence_expr_as_boolean_expr_negative_delay(self):
        m = self.get_model()
        e = m.b.implies(m.i2[2].start_time.at(m.i2[1].start_time, delay=-3))
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))
        self.assertIn(id(m.b), visitor.var_map)
        self.assertIn(id(m.i2[1]), visitor.var_map)
        self.assertIn(id(m.i2[2]), visitor.var_map)
        b = visitor.var_map[id(m.b)]
        i21 = visitor.var_map[id(m.i2[1])]
        i22 = visitor.var_map[id(m.i2[2])]
        self.assertTrue(expr[1].equals(cp.if_then(b, cp.start_of(i22) + -3 == cp.start_of(i21))))