import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
@unittest.skipUnless(sympy_available, 'Sympy not available')
class TestLogicalToLinearTransformation(unittest.TestCase):

    def test_longer_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(lor(m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(1, m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary() + (1 - m.Y[1].get_associated_binary()), None)], m.logic_to_linear.transformed_constraints)

    def test_xfrm_atleast_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=atleast(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(2, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), None)], m.logic_to_linear.transformed_constraints)

    def test_xfrm_atmost_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=atmost(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(None, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), 2)], m.logic_to_linear.transformed_constraints)

    def test_xfrm_exactly_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=exactly(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(2, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), 2)], m.logic_to_linear.transformed_constraints)

    def test_xfrm_special_atoms_nonroot(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(atleast(2, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 1)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(self, [(None, sum(m.Y[:].get_associated_binary()) - (1 + 2 * Y_aug[1].get_associated_binary()), 0), (1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, 2 - 2 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(atmost(2, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 1)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(self, [(None, sum(m.Y[:].get_associated_binary()) - (1 - Y_aug[1].get_associated_binary() + 2), 0), (1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, 3 - 3 * Y_aug[1].get_associated_binary() - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalConstraint(expr=m.Y[1].implies(exactly(2, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 3)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, sum(m.Y[:].get_associated_binary()) - (1 - Y_aug[1].get_associated_binary() + 2), 0), (None, 2 - 2 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0), (1, sum(Y_aug[:].get_associated_binary()), None), (None, sum(m.Y[:].get_associated_binary()) - (1 + 2 * (1 - Y_aug[2].get_associated_binary())), 0), (None, 3 - 3 * (1 - Y_aug[3].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.x = Var(bounds=(1, 3))
        m.p = LogicalConstraint(expr=m.Y[1].implies(exactly(m.x, m.Y[1], m.Y[2], m.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 3)
        self.assertEqual(Y_aug[1].domain, BooleanSet)
        _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, sum(m.Y[:].get_associated_binary()) - (m.x + 2 * (1 - Y_aug[1].get_associated_binary())), 0), (None, m.x - 3 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0), (1, sum(Y_aug[:].get_associated_binary()), None), (None, sum(m.Y[:].get_associated_binary()) - (m.x - 1 + 3 * (1 - Y_aug[2].get_associated_binary())), 0), (None, m.x + 1 - 4 * (1 - Y_aug[3].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)

    def test_xfrm_atleast_nested(self):
        m = _generate_boolean_model(4)
        m.p = LogicalConstraint(expr=atleast(1, atleast(2, m.Y[1], m.Y[1].lor(m.Y[2]), m.Y[2]).lor(m.Y[3]), m.Y[4]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        Y_aug = m.logic_to_linear.augmented_vars
        self.assertEqual(len(Y_aug), 3)
        _constrs_contained_within(self, [(1, Y_aug[1].get_associated_binary() + m.Y[4].get_associated_binary(), None), (1, 1 - Y_aug[2].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (1, 1 - m.Y[3].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (1, Y_aug[2].get_associated_binary() + m.Y[3].get_associated_binary() + 1 - Y_aug[1].get_associated_binary(), None), (1, 1 - m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary(), None), (1, 1 - m.Y[2].get_associated_binary() + Y_aug[3].get_associated_binary(), None), (1, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + 1 - Y_aug[3].get_associated_binary(), None), (None, 2 - 2 * (1 - Y_aug[2].get_associated_binary()) - (m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary() + m.Y[2].get_associated_binary()), 0), (None, m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary() + m.Y[2].get_associated_binary() - (1 + 2 * Y_aug[2].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)

    def test_link_with_gdp_indicators(self):
        m = _generate_boolean_model(4)
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.x = Var()
        m.dd = Disjunct([1, 2])
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d2.c = Constraint(expr=m.x <= 10)
        m.dd[1].c = Constraint(expr=m.x >= 5)
        m.dd[2].c = Constraint(expr=m.x <= 6)
        m.Y[1].associate_binary_var(m.d1.binary_indicator_var)
        m.Y[2].associate_binary_var(m.d2.binary_indicator_var)
        m.Y[3].associate_binary_var(m.dd[1].binary_indicator_var)
        m.Y[4].associate_binary_var(m.dd[2].binary_indicator_var)
        m.p = LogicalConstraint(expr=m.Y[1].implies(lor(m.Y[3], m.Y[4])))
        m.p2 = LogicalConstraint(expr=atmost(2, *m.Y[:]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        _constrs_contained_within(self, [(1, m.dd[1].binary_indicator_var + m.dd[2].binary_indicator_var + 1 - m.d1.binary_indicator_var, None), (None, m.d1.binary_indicator_var + m.d2.binary_indicator_var + m.dd[1].binary_indicator_var + m.dd[2].binary_indicator_var, 2)], m.logic_to_linear.transformed_constraints)

    def test_gdp_nesting(self):
        m = _generate_boolean_model(2)
        m.disj = Disjunction(expr=[[m.Y[1].implies(m.Y[2])], [m.Y[2].equivalent_to(False)]])
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=[m.disj.disjuncts[0], m.disj.disjuncts[1]])
        _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary(), None)], m.disj_disjuncts[0].logic_to_linear.transformed_constraints)
        _constrs_contained_within(self, [(1, 1 - m.Y[2].get_associated_binary(), 1)], m.disj_disjuncts[1].logic_to_linear.transformed_constraints)

    def test_transformed_components_on_parent_block(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.s = RangeSet(3)
        m.b.Y = BooleanVar(m.b.s)
        m.b.p = LogicalConstraint(expr=m.b.Y[1].implies(lor(m.b.Y[2], m.b.Y[3])))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        boolean_var = m.b.component('Y_asbinary')
        self.assertIsInstance(boolean_var, Var)
        notAVar = m.component('Y_asbinary')
        self.assertIsNone(notAVar)
        transBlock = m.b.component('logic_to_linear')
        self.assertIsInstance(transBlock, Block)
        notAThing = m.component('logic_to_linear')
        self.assertIsNone(notAThing)
        _constrs_contained_within(self, [(1, m.b.Y[2].get_associated_binary() + m.b.Y[3].get_associated_binary() + (1 - m.b.Y[1].get_associated_binary()), None)], m.b.logic_to_linear.transformed_constraints)

    def make_nested_block_model(self):
        """For the next two tests: Has BooleanVar on model, but
        LogicalConstraints on a Block and a Block nested on that Block."""
        m = ConcreteModel()
        m.b = Block()
        m.Y = BooleanVar([1, 2])
        m.b.logical = LogicalConstraint(expr=~m.Y[1])
        m.b.b = Block()
        m.b.b.logical = LogicalConstraint(expr=m.Y[1].xor(m.Y[2]))
        return m

    def test_transform_block(self):
        m = self.make_nested_block_model()
        TransformationFactory('core.logical_to_linear').apply_to(m.b)
        _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary(), 1)], m.b.logic_to_linear.transformed_constraints)
        _constrs_contained_within(self, [(1, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary(), None), (1, 1 - m.Y[1].get_associated_binary() + 1 - m.Y[2].get_associated_binary(), None)], m.b.b.logic_to_linear.transformed_constraints)
        self.assertEqual(len(m.b.logic_to_linear.transformed_constraints), 1)
        self.assertEqual(len(m.b.b.logic_to_linear.transformed_constraints), 2)

    def test_transform_targets_on_block(self):
        m = self.make_nested_block_model()
        TransformationFactory('core.logical_to_linear').apply_to(m.b, targets=m.b.b)
        self.assertIsNone(m.b.component('logic_to_linear'))
        _constrs_contained_within(self, [(1, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary(), None), (1, 1 - m.Y[1].get_associated_binary() + 1 - m.Y[2].get_associated_binary(), None)], m.b.b.logic_to_linear.transformed_constraints)
        self.assertEqual(len(m.b.b.logic_to_linear.transformed_constraints), 2)

    def test_logical_constraint_target(self):
        m = _generate_boolean_model(3)
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.constraint)
        _constrs_contained_within(self, [(2, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), 2)], m.logic_to_linear.transformed_constraints)

    def make_indexed_logical_constraint_model(self):
        m = _generate_boolean_model(3)
        m.cons = LogicalConstraint([1, 2])
        m.cons[1] = exactly(2, m.Y)
        m.cons[2] = m.Y[1].implies(lor(m.Y[2], m.Y[3]))
        return m

    def test_indexed_logical_constraint_target(self):
        m = self.make_indexed_logical_constraint_model()
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.cons)
        _constrs_contained_within(self, [(2, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary(), 2)], m.logic_to_linear.transformed_constraints)
        _constrs_contained_within(self, [(1, m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary() + (1 - m.Y[1].get_associated_binary()), None)], m.logic_to_linear.transformed_constraints)
        self.assertEqual(len(m.logic_to_linear.transformed_constraints), 2)
        self.assertTrue(m.constraint.active)

    def test_logical_constraintData_target(self):
        m = self.make_indexed_logical_constraint_model()
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.cons[2])
        _constrs_contained_within(self, [(1, m.Y[2].get_associated_binary() + m.Y[3].get_associated_binary() + (1 - m.Y[1].get_associated_binary()), None)], m.logic_to_linear.transformed_constraints)
        self.assertEqual(len(m.logic_to_linear.transformed_constraints), 1)

    def test_blockData_target(self):
        m = ConcreteModel()
        m.b = Block([1, 2])
        m.b[1].transfer_attributes_from(self.make_indexed_logical_constraint_model())
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.b[1])
        _constrs_contained_within(self, [(2, m.b[1].Y[1].get_associated_binary() + m.b[1].Y[2].get_associated_binary() + m.b[1].Y[3].get_associated_binary(), 2)], m.b[1].logic_to_linear.transformed_constraints)
        _constrs_contained_within(self, [(1, m.b[1].Y[2].get_associated_binary() + m.b[1].Y[3].get_associated_binary() + (1 - m.b[1].Y[1].get_associated_binary()), None)], m.b[1].logic_to_linear.transformed_constraints)

    def test_disjunctData_target(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2])
        m.d[1].transfer_attributes_from(self.make_indexed_logical_constraint_model())
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=m.d[1])
        _constrs_contained_within(self, [(2, m.d[1].Y[1].get_associated_binary() + m.d[1].Y[2].get_associated_binary() + m.d[1].Y[3].get_associated_binary(), 2)], m.d[1].logic_to_linear.transformed_constraints)
        _constrs_contained_within(self, [(1, m.d[1].Y[2].get_associated_binary() + m.d[1].Y[3].get_associated_binary() + (1 - m.d[1].Y[1].get_associated_binary()), None)], m.d[1].logic_to_linear.transformed_constraints)

    def test_target_with_unrecognized_type(self):
        m = _generate_boolean_model(2)
        with self.assertRaisesRegex(ValueError, "invalid value for configuration 'targets':\\n\\tFailed casting 1\\n\\tto target_list\\n\\tError: Expected Component or list of Components.\\n\\tReceived <class 'int'>"):
            TransformationFactory('core.logical_to_linear').apply_to(m, targets=1)

    def test_mixed_logical_relational_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = BooleanVar([1, 2])
        m.c = LogicalConstraint(expr=land(m.y[1], m.y[2]).implies(m.x >= 0))
        with self.assertRaisesRegex(MouseTrap, "core.logical_to_linear does not support transforming LogicalConstraints with embedded relational expressions. Found '0.0 <= x'.", normalize_whitespace=True):
            TransformationFactory('core.logical_to_linear').apply_to(m)

    def test_external_function(self):

        def _fcn(*args):
            raise RuntimeError('unreachable')
        m = ConcreteModel()
        m.x = Var()
        m.f = ExternalFunction(_fcn)
        m.y = BooleanVar()
        m.c = LogicalConstraint(expr=m.y.implies(m.f(m.x)))
        with self.assertRaisesRegex(TypeError, "Expressions containing external functions are not convertible to sympy expressions \\(found 'f\\(x1"):
            TransformationFactory('core.logical_to_linear').apply_to(m)