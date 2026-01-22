from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
class TestLogicalToDisjunctiveTransformation(unittest.TestCase):

    def make_model(self):
        m = ConcreteModel()
        m.a = BooleanVar()
        m.b = BooleanVar([1, 2])
        m.p = Param(initialize=1)
        m.p2 = Param([1, 2], mutable=True)
        m.p2[1] = 1
        m.p2[2] = 2
        m.block = Block()
        m.block.c1 = LogicalConstraint(expr=m.a.land(m.b[1]))
        m.block.c2 = LogicalConstraint(expr=exactly(m.p2[2], m.a, m.b[1], m.b[2].lor(m.b[1])))
        m.c1 = LogicalConstraint(expr=atmost(m.p + m.p2[1], m.a, m.b[1], m.b[2]))
        return m

    def check_and_constraints(self, a, b1, z, transBlock):
        assertExpressionsEqual(self, transBlock.transformed_constraints[1].expr, z <= a)
        assertExpressionsEqual(self, transBlock.transformed_constraints[2].expr, z <= b1)
        assertExpressionsEqual(self, transBlock.transformed_constraints[3].expr, 1 - z <= 2 - (a + b1))
        assertExpressionsEqual(self, transBlock.transformed_constraints[4].expr, z >= 1)

    def check_block_c1_transformed(self, m, transBlock):
        self.assertFalse(m.block.c1.active)
        self.assertIs(m.a.get_associated_binary(), transBlock.auxiliary_vars[1])
        self.assertIs(m.b[1].get_associated_binary(), transBlock.auxiliary_vars[2])
        self.check_and_constraints(transBlock.auxiliary_vars[1], transBlock.auxiliary_vars[2], transBlock.auxiliary_vars[3], transBlock)

    def check_block_exactly(self, a, b1, b2, z4, transBlock):
        m = transBlock.model()
        assertExpressionsEqual(self, transBlock.transformed_constraints[5].expr, 1 - z4 + b2 + b1 >= 1)
        assertExpressionsEqual(self, transBlock.transformed_constraints[6].expr, z4 + (1 - b2) >= 1)
        assertExpressionsEqual(self, transBlock.transformed_constraints[7].expr, z4 + (1 - b1) >= 1)
        assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[0].constraint.expr, a + b1 + z4 == m.p2[2])
        assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[1].disjunction.disjuncts[0].constraint[1].expr, a + b1 + z4 <= m.p2[2] - 1)
        assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[1].disjunction.disjuncts[1].constraint[1].expr, a + b1 + z4 >= m.p2[2] + 1)
        assertExpressionsEqual(self, transBlock.transformed_constraints[8].expr, transBlock.auxiliary_disjuncts[0].binary_indicator_var >= 1)

    def check_block_transformed(self, m):
        self.assertFalse(m.block.c2.active)
        transBlock = m.block._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 5)
        self.assertEqual(len(transBlock.transformed_constraints), 8)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
        self.check_block_c1_transformed(m, transBlock)
        self.assertIs(m.b[2].get_associated_binary(), transBlock.auxiliary_vars[4])
        z4 = transBlock.auxiliary_vars[5]
        a = transBlock.auxiliary_vars[1]
        b1 = transBlock.auxiliary_vars[2]
        b2 = transBlock.auxiliary_vars[4]
        self.check_block_exactly(a, b1, b2, z4, transBlock)

    def test_constraint_target(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m, targets=[m.block.c1])
        transBlock = m.block._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 3)
        self.assertEqual(len(transBlock.transformed_constraints), 4)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 0)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 0)
        self.check_block_c1_transformed(m, transBlock)
        self.assertTrue(m.block.c2.active)
        self.assertTrue(m.c1.active)

    def test_block_target(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m, targets=[m.block])
        self.check_block_transformed(m)
        self.assertTrue(m.c1.active)

    def test_transform_block(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m.block)
        self.check_block_transformed(m)
        self.assertTrue(m.c1.active)

    def test_transform_model(self):
        m = self.make_model()
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m)
        self.assertFalse(m.c1.active)
        transBlock = m._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 3)
        self.assertEqual(len(transBlock.transformed_constraints), 1)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
        a = m._logical_to_disjunctive.auxiliary_vars[1]
        b1 = m._logical_to_disjunctive.auxiliary_vars[2]
        b2 = m._logical_to_disjunctive.auxiliary_vars[3]
        assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[0].constraint.expr, a + b1 + b2 <= 1 + m.p2[1])
        assertExpressionsEqual(self, transBlock.auxiliary_disjuncts[1].constraint.expr, a + b1 + b2 >= 1 + m.p2[1] + 1)
        assertExpressionsEqual(self, transBlock.transformed_constraints[1].expr, transBlock.auxiliary_disjuncts[0].binary_indicator_var >= 1)
        transBlock = m.block._logical_to_disjunctive
        self.assertEqual(len(transBlock.auxiliary_vars), 2)
        self.assertEqual(len(transBlock.transformed_constraints), 8)
        self.assertEqual(len(transBlock.auxiliary_disjuncts), 2)
        self.assertEqual(len(transBlock.auxiliary_disjunctions), 1)
        self.check_and_constraints(a, b1, transBlock.auxiliary_vars[1], transBlock)
        self.check_block_exactly(a, b1, b2, transBlock.auxiliary_vars[2], transBlock)

    @unittest.skipUnless(gurobi_available, 'Gurobi is not available')
    def test_reverse_implication_for_land(self):
        m = ConcreteModel()
        m.t = BooleanVar()
        m.a = BooleanVar()
        m.d = BooleanVar()
        m.c = LogicalConstraint(expr=m.t.equivalent_to(m.a.land(m.d)))
        m.a.fix(True)
        m.d.fix(True)
        m.binary = Var(domain=Binary)
        m.t.associate_binary_var(m.binary)
        m.obj = Objective(expr=m.binary)
        TransformationFactory('contrib.logical_to_disjunctive').apply_to(m)
        TransformationFactory('gdp.bigm').apply_to(m)
        SolverFactory('gurobi').solve(m)
        update_boolean_vars_from_binary(m)
        self.assertEqual(value(m.obj), 1)
        self.assertTrue(value(m.t))