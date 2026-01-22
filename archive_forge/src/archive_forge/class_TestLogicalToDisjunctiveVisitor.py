from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
class TestLogicalToDisjunctiveVisitor(unittest.TestCase):

    def make_model(self):
        m = ConcreteModel()
        m.a = BooleanVar()
        m.b = BooleanVar()
        m.c = BooleanVar()
        return m

    def test_logical_or(self):
        m = self.make_model()
        e = lor(m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        self.assertEqual(len(m.cons), 5)
        self.assertEqual(len(m.z), 4)
        assertExpressionsEqual(self, m.cons[1].expr, 1 - m.z[4] + m.z[1] + m.z[2] + m.z[3] >= 1)
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] + (1 - m.z[1]) >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, m.z[4] + (1 - m.z[2]) >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, m.z[4] + (1 - m.z[3]) >= 1)
        assertExpressionsEqual(self, m.cons[5].expr, m.z[4] >= 1)

    def test_logical_and(self):
        m = self.make_model()
        e = land(m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        self.assertEqual(len(m.cons), 5)
        self.assertEqual(len(m.z), 4)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[4] <= m.z[1])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(self, m.cons[3].expr, m.z[4] <= m.z[3])
        assertExpressionsEqual(self, m.cons[4].expr, 1 - m.z[4] <= 3 - (m.z[1] + m.z[2] + m.z[3]))
        assertExpressionsEqual(self, m.cons[5].expr, m.z[4] >= 1)

    def test_logical_not(self):
        m = self.make_model()
        e = lnot(m.a)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 2)
        self.assertEqual(len(m.z), 2)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[2] == 1 - m.z[1])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= 1)

    def test_implication(self):
        m = self.make_model()
        e = m.a.implies(m.b.land(m.c))
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        self.assertEqual(len(m.cons), 7)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[4] <= m.z[2])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[4] <= m.z[3])
        assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[4] <= 2 - (m.z[2] + m.z[3]))
        assertExpressionsEqual(self, m.cons[4].expr, 1 - m.z[5] + (1 - m.z[1]) + m.z[4] >= 1)
        assertExpressionsEqual(self, m.cons[5].expr, m.z[5] + (1 - (1 - m.z[1])) >= 1)
        assertExpressionsEqual(self, m.cons[6].expr, m.z[5] + (1 - m.z[4]) >= 1)
        assertExpressionsEqual(self, m.cons[7].expr, m.z[5] >= 1)

    def test_equivalence(self):
        m = self.make_model()
        e = m.a.equivalent_to(m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.c.get_associated_binary(), m.z[2])
        self.assertEqual(len(m.z), 5)
        self.assertEqual(len(m.cons), 10)
        assertExpressionsEqual(self, m.cons[1].expr, 1 - m.z[3] + (1 - m.z[1]) + m.z[2] >= 1)
        assertExpressionsEqual(self, m.cons[2].expr, 1 - (1 - m.z[1]) + m.z[3] >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, m.z[3] + (1 - m.z[2]) >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, 1 - m.z[4] + (1 - m.z[2]) + m.z[1] >= 1)
        assertExpressionsEqual(self, m.cons[5].expr, m.z[4] + (1 - m.z[1]) >= 1)
        assertExpressionsEqual(self, m.cons[6].expr, 1 - (1 - m.z[2]) + m.z[4] >= 1)
        assertExpressionsEqual(self, m.cons[7].expr, m.z[5] <= m.z[3])
        assertExpressionsEqual(self, m.cons[8].expr, m.z[5] <= m.z[4])
        assertExpressionsEqual(self, m.cons[9].expr, 1 - m.z[5] <= 2 - (m.z[3] + m.z[4]))
        assertExpressionsEqual(self, m.cons[10].expr, m.z[5] >= 1)

    def test_xor(self):
        m = self.make_model()
        e = m.a.xor(m.b)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        self.assertEqual(len(m.z), 2)
        self.assertEqual(len(m.cons), 1)
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(m.disjunctions), 1)
        assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] == 1)
        assertExpressionsEqual(self, m.disjuncts[1].disjunction.disjuncts[0].constraint[1].expr, m.z[1] + m.z[2] <= 0)
        assertExpressionsEqual(self, m.disjuncts[1].disjunction.disjuncts[1].constraint[1].expr, m.z[1] + m.z[2] >= 2)
        assertExpressionsEqual(self, m.cons[1].expr, m.disjuncts[0].binary_indicator_var >= 1)

    def test_at_most(self):
        m = self.make_model()
        e = atmost(2, m.a, m.a.land(m.b), m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[4])
        c = m.z[4]
        self.assertEqual(len(m.z), 4)
        self.assertEqual(len(m.cons), 4)
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(m.disjunctions), 1)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[3] <= a)
        assertExpressionsEqual(self, m.cons[2].expr, m.z[3] <= b)
        m.cons.pprint()
        print(m.cons[3].expr)
        assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[3] <= 2 - sum([a, b]))
        assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[3] + m.z[4] <= 2)
        assertExpressionsEqual(self, m.disjuncts[1].constraint.expr, m.z[1] + m.z[3] + m.z[4] >= 3)
        assertExpressionsEqual(self, m.cons[4].expr, m.disjuncts[0].binary_indicator_var >= 1)

    def test_at_least(self):
        m = self.make_model()
        e = atleast(2, m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        a = m.z[1]
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        b = m.z[2]
        self.assertIs(m.c.get_associated_binary(), m.z[3])
        c = m.z[3]
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 1)
        assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] + m.z[3] >= 2)
        assertExpressionsEqual(self, m.disjuncts[1].constraint.expr, m.z[1] + m.z[2] + m.z[3] <= 1)
        assertExpressionsEqual(self, m.cons[1].expr, m.disjuncts[0].binary_indicator_var >= 1)

    @unittest.skipUnless(gurobi_available, 'Gurobi is not available')
    def test_logical_integration(self):
        """
        This is kind of a ridiculous test, but I bothered type it and it has
        a lot of logical things together, so adding it.
        """
        m = self.make_model()
        m.d = BooleanVar()
        m.t = BooleanVar()
        e = m.t.equivalent_to(lnot(lor(m.a, m.b)).land(exactly(1, [m.c, m.d])))
        m.c.fix(True)
        m.d.fix(False)
        m.a.fix(False)
        m.b.fix(False)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 11)
        self.assertIs(m.a.get_associated_binary(), m.z[2])
        a = m.z[2]
        self.assertIs(m.b.get_associated_binary(), m.z[3])
        b = m.z[3]
        z3 = m.z[4]
        z4 = m.z[5]
        z5 = m.z[8]
        self.assertIs(m.t.get_associated_binary(), m.z[1])
        t = m.z[1]
        z6 = m.z[9]
        z7 = m.z[10]
        self.assertIs(m.c.get_associated_binary(), m.z[6])
        c = m.z[6]
        self.assertIs(m.d.get_associated_binary(), m.z[7])
        d = m.z[7]
        z8 = m.z[11]
        self.assertEqual(len(m.disjuncts), 2)
        self.assertEqual(len(list(m.disjuncts[0].component_data_objects(Constraint, descend_into=False))), 1)
        assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, c + d == 1)
        self.assertEqual(len(list(m.disjuncts[0].component_data_objects(Disjunct, descend_into=False))), 0)
        self.assertEqual(len(list(m.disjuncts[1].component_data_objects(Constraint, descend_into=False))), 0)
        self.assertEqual(len(list(m.disjuncts[1].component_data_objects(Disjunct, descend_into=False))), 2)
        self.assertEqual(len(m.disjuncts[1].disjunction.disjuncts), 2)
        assertExpressionsEqual(self, m.disjuncts[1].disjunction.disjuncts[0].constraint[1].expr, c + d <= 0)
        assertExpressionsEqual(self, m.disjuncts[1].disjunction.disjuncts[1].constraint[1].expr, c + d >= 2)
        self.assertEqual(len(m.disjunctions), 1)
        self.assertIs(m.disjunctions[0].disjuncts[0], m.disjuncts[0])
        self.assertIs(m.disjunctions[0].disjuncts[1], m.disjuncts[1])
        self.assertEqual(len(m.cons), 17)
        assertExpressionsEqual(self, m.cons[1].expr, 1 - z3 + a + b >= 1)
        assertExpressionsEqual(self, m.cons[2].expr, z3 + (1 - a) >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, z3 + (1 - b) >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, z4 == 1 - z3)
        assertExpressionsEqual(self, m.cons[5].expr, z5 <= z4)
        assertExpressionsEqual(self, m.cons[6].expr, z5 <= m.disjuncts[0].binary_indicator_var)
        assertExpressionsEqual(self, m.cons[7].expr, 1 - z5 <= 2 - (z4 + m.disjuncts[0].binary_indicator_var))
        assertExpressionsEqual(self, m.cons[8].expr, 1 - z6 + (1 - t) + z5 >= 1)
        assertExpressionsEqual(self, m.cons[9].expr, 1 - (1 - t) + z6 >= 1)
        assertExpressionsEqual(self, m.cons[10].expr, z6 + (1 - z5) >= 1)
        assertExpressionsEqual(self, m.cons[11].expr, 1 - z7 + (1 - z5) + t >= 1)
        assertExpressionsEqual(self, m.cons[12].expr, z7 + (1 - t) >= 1)
        assertExpressionsEqual(self, m.cons[13].expr, 1 - (1 - z5) + z7 >= 1)
        assertExpressionsEqual(self, m.cons[14].expr, z8 <= z6)
        assertExpressionsEqual(self, m.cons[15].expr, z8 <= z7)
        assertExpressionsEqual(self, m.cons[16].expr, 1 - z8 <= 2 - (z6 + z7))
        assertExpressionsEqual(self, m.cons[17].expr, z8 >= 1)
        TransformationFactory('gdp.bigm').apply_to(m)
        m.obj = Objective(expr=m.t.get_associated_binary())
        SolverFactory('gurobi').solve(m, tee=True)
        update_boolean_vars_from_binary(m)
        self.assertTrue(value(e))
        self.assertEqual(value(m.obj), 1)
        self.assertTrue(value(m.t))

    def test_boolean_fixed_true(self):
        m = self.make_model()
        e = m.a.implies(m.b)
        m.a.fix(True)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 4)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertTrue(m.z[1].fixed)
        self.assertEqual(value(m.z[1]), 1)
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        assertExpressionsEqual(self, m.cons[1].expr, 1 - m.z[3] + (1 - m.z[1]) + m.z[2] >= 1)
        assertExpressionsEqual(self, m.cons[2].expr, 1 - (1 - m.z[1]) + m.z[3] >= 1)
        assertExpressionsEqual(self, m.cons[3].expr, m.z[3] + (1 - m.z[2]) >= 1)
        assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)

    def test_boolean_fixed_false(self):
        m = self.make_model()
        e = m.a & m.b
        m.a.fix(False)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 4)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertTrue(m.z[1].fixed)
        self.assertEqual(value(m.z[1]), 0)
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= m.z[3])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= m.z[3])
        assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[3] <= 2 - (m.z[1] + m.z[2]))
        assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)

    def test_boolean_fixed_none(self):
        m = self.make_model()
        e = m.a & m.b
        m.a.fix(None)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        m.disjuncts = visitor.disjuncts
        m.disjunctions = visitor.disjunctions
        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 3)
        self.assertEqual(len(m.cons), 4)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertTrue(m.z[1].fixed)
        self.assertIsNone(m.z[1].value)
        self.assertIs(m.b.get_associated_binary(), m.z[2])
        assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= m.z[3])
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= m.z[3])
        assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[3] <= 2 - (m.z[1] + m.z[2]))
        assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)

    def test_no_need_to_walk(self):
        m = self.make_model()
        e = m.a
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 1)
        self.assertIs(m.a.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 1)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= 1)

    def test_binary_already_associated(self):
        m = self.make_model()
        m.mine = Var(domain=Binary)
        m.a.associate_binary_var(m.mine)
        e = m.a.land(m.b)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        visitor.walk_expression(e)
        self.assertEqual(len(m.z), 2)
        self.assertIs(m.b.get_associated_binary(), m.z[1])
        self.assertEqual(len(m.cons), 4)
        assertExpressionsEqual(self, m.cons[1].expr, m.z[2] <= m.mine)
        assertExpressionsEqual(self, m.cons[2].expr, m.z[2] <= m.z[1])
        assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[2] <= 2 - (m.mine + m.z[1]))
        assertExpressionsEqual(self, m.cons[4].expr, m.z[2] >= 1)

    def test_integer_var_in_at_least(self):
        m = self.make_model()
        m.x = Var(bounds=(0, 10), domain=Integers)
        e = atleast(m.x, m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        with self.assertRaisesRegex(MouseTrap, "The first argument 'x' to 'atleast\\(x: \\[a, b, c\\]\\)' is potentially variable. This may be a mathematically coherent expression; However it is not yet supported to convert it to a disjunctive program.", normalize_whitespace=True):
            visitor.walk_expression(e)

    def test_numeric_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
        m.y = Var(domain=Integers)
        m.e = Expression(expr=m.x[1] * m.x[2])
        e = atmost(m.e + m.y, m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        with self.assertRaisesRegex(MouseTrap, "The first argument '\\(x\\[1\\]\\*x\\[2\\]\\) \\+ y' to 'atmost\\(\\(x\\[1\\]\\*x\\[2\\]\\) \\+ y: \\[a, b, c\\]\\)' is potentially variable. This may be a mathematically coherent expression; However it is not yet supported to convert it to a disjunctive program", normalize_whitespace=True):
            visitor.walk_expression(e)

    def test_named_expression_in_at_most(self):
        m = self.make_model()
        m.x = Var([1, 2], bounds=(0, 10), domain=Integers)
        m.y = Var(domain=Integers)
        m.e = Expression(expr=m.x[1] * m.x[2])
        e = atmost(m.e, m.a, m.b, m.c)
        visitor = LogicalToDisjunctiveVisitor()
        m.cons = visitor.constraints
        m.z = visitor.z_vars
        with self.assertRaisesRegex(MouseTrap, "The first argument 'e' to 'atmost\\(\\(x\\[1\\]\\*x\\[2\\]\\): \\[a, b, c\\]\\)' is potentially variable. This may be a mathematically coherent expression; However it is not yet supported to convert it to a disjunctive program", normalize_whitespace=True):
            visitor.walk_expression(e)

    def test_relational_expr_as_boolean_atom(self):
        m = self.make_model()
        m.x = Var()
        e = m.a.land(m.x >= 3)
        visitor = LogicalToDisjunctiveVisitor()
        with self.assertRaisesRegex(MouseTrap, "The RelationalExpression '3 <= x' was used as a Boolean term in a logical proposition. This is not yet supported when transforming to disjunctive form.", normalize_whitespace=True):
            visitor.walk_expression(e)