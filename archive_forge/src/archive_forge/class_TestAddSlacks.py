import os
from os.path import abspath, dirname
from io import StringIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
import random
from pyomo.opt import check_available_solvers
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.compare import assertExpressionsEqual
class TestAddSlacks(unittest.TestCase):

    def setUp(self):
        random.seed(666)

    @staticmethod
    def makeModel():
        model = ConcreteModel()
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.rule1 = Constraint(expr=model.x <= 5)
        model.rule2 = Constraint(expr=inequality(1, model.y, 3))
        model.rule3 = Constraint(expr=model.x >= 0.1)
        model.obj = Objective(expr=-model.x - model.y)
        return model

    def test_add_trans_block(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        xblock = m.component('_core_add_slack_variables')
        self.assertIsInstance(xblock, Block)

    def test_trans_block_name_collision(self):
        m = self.makeModel()
        m._core_add_slack_variables = Block()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        xblock = m.component('_core_add_slack_variables_4')
        self.assertIsInstance(xblock, Block)

    def test_slack_vars_added(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        xblock = m.component('_core_add_slack_variables')
        self.assertIsInstance(xblock.component('_slack_minus_rule1'), Var)
        self.assertFalse(hasattr(xblock, '_slack_plus_rule1'))
        self.assertIsInstance(xblock.component('_slack_minus_rule2'), Var)
        self.assertIsInstance(xblock.component('_slack_plus_rule2'), Var)
        self.assertFalse(hasattr(xblock, '_slack_minus_rule3'))
        self.assertIsInstance(xblock.component('_slack_plus_rule3'), Var)
        self.assertEqual(xblock._slack_minus_rule1.bounds, (0, None))
        self.assertEqual(xblock._slack_minus_rule2.bounds, (0, None))
        self.assertEqual(xblock._slack_plus_rule2.bounds, (0, None))
        self.assertEqual(xblock._slack_plus_rule3.bounds, (0, None))

    def checkRule1(self, m):
        cons = m.rule1
        transBlock = m.component('_core_add_slack_variables')
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 5)
        assertExpressionsEqual(self, cons.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.x)), EXPR.MonomialTermExpression((-1, transBlock._slack_minus_rule1))]))

    def checkRule3(self, m):
        cons = m.rule3
        transBlock = m.component('_core_add_slack_variables')
        self.assertIsNone(cons.upper)
        self.assertEqual(cons.lower, 0.1)
        assertExpressionsEqual(self, cons.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.x)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule3))]))

    def test_ub_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.checkRule1(m)

    def test_lb_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.checkRule3(m)

    def test_both_bounds_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        cons = m.rule2
        transBlock = m.component('_core_add_slack_variables')
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        assertExpressionsEqual(self, cons.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.y)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule2)), EXPR.MonomialTermExpression((-1, transBlock._slack_minus_rule2))]))

    def test_obj_deactivated(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.assertFalse(m.obj.active)

    def test_new_obj_created(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        transBlock = m.component('_core_add_slack_variables')
        obj = transBlock.component('_slack_objective')
        self.assertIsInstance(obj, Objective)
        self.assertTrue(obj.active)
        assertExpressionsEqual(self, obj.expr, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, transBlock._slack_minus_rule1)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule2)), EXPR.MonomialTermExpression((1, transBlock._slack_minus_rule2)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule3))]))

    def test_badModel_err(self):
        model = ConcreteModel()
        model.x = Var(within=NonNegativeReals)
        model.rule1 = Constraint(expr=inequality(6, model.x, 5))
        self.assertRaisesRegex(RuntimeError, 'Lower bound exceeds upper bound in constraint rule1*', TransformationFactory('core.add_slack_variables').apply_to, model)

    def test_leave_deactivated_constraints(self):
        m = self.makeModel()
        m.rule2.deactivate()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        cons = m.rule2
        self.assertFalse(cons.active)
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        self.assertIs(cons.body, m.y)

    def checkTargetSlackVar(self, transBlock):
        self.assertIsNone(transBlock.component('_slack_minus_rule2'))
        self.assertIsNone(transBlock.component('_slack_plus_rule2'))
        self.assertFalse(hasattr(transBlock, '_slack_minus_rule3'))
        self.assertIsInstance(transBlock.component('_slack_plus_rule3'), Var)

    def checkTargetSlackVars(self, transBlock):
        self.assertIsInstance(transBlock.component('_slack_minus_rule1'), Var)
        self.assertFalse(hasattr(transBlock, '_slack_plus_rule1'))
        self.assertIsNone(transBlock.component('_slack_minus_rule2'))
        self.assertIsNone(transBlock.component('_slack_plus_rule2'))
        self.assertFalse(hasattr(transBlock, '_slack_minus_rule3'))
        self.assertIsInstance(transBlock.component('_slack_plus_rule3'), Var)

    def test_only_targets_have_slack_vars(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1, m.rule3])
        transBlock = m.component('_core_add_slack_variables')
        self.checkTargetSlackVars(transBlock)

    def test_only_targets_have_slack_vars_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1, m.rule3])
        transBlock = m2.component('_core_add_slack_variables')
        self.checkTargetSlackVars(transBlock)

    def checkNonTargetCons(self, m):
        cons = m.rule2
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        self.assertIs(cons.body, m.y)

    def test_nontarget_constraint_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1, m.rule3])
        self.checkNonTargetCons(m)

    def test_nontarget_constraint_same_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1, m.rule3])
        self.checkNonTargetCons(m2)

    def test_target_constraints_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1, m.rule3])
        self.checkRule1(m)
        self.checkRule3(m)

    def test_target_constraints_transformed_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1, m.rule3])
        self.checkRule1(m2)
        self.checkRule3(m2)

    def checkTargetObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component('_slack_objective')
        self.assertIs(obj.expr, transBlock._slack_plus_rule3)

    def checkTargetsObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component('_slack_objective')
        assertExpressionsEqual(self, obj.expr, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, transBlock._slack_minus_rule1)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule3))]))

    def test_target_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1, m.rule3])
        self.assertFalse(m.obj.active)
        self.checkTargetsObj(m)

    def test_target_objective_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1, m.rule3])
        self.assertFalse(m2.obj.active)
        self.checkTargetsObj(m2)

    def test_err_for_bogus_kwds(self):
        m = self.makeModel()
        self.assertRaisesRegex(ValueError, "key 'notakwd' not defined for ConfigDict ''", TransformationFactory('core.add_slack_variables').apply_to, m, notakwd='I want a feasible model')

    def test_error_for_non_constraint_noniterable_target(self):
        m = self.makeModel()
        m.indexedVar = Var([1, 2])
        self.assertRaisesRegex(ValueError, "Expected Constraint or list of Constraints.\n\tReceived <class 'pyomo.core.base.var._GeneralVarData'>", TransformationFactory('core.add_slack_variables').apply_to, m, targets=m.indexedVar[1])

    def test_error_for_non_constraint_target_in_list(self):
        m = self.makeModel()
        self.assertRaisesRegex(ValueError, "Expected Constraint or list of Constraints.\n\tReceived <class 'pyomo.core.base.var.ScalarVar'>", TransformationFactory('core.add_slack_variables').apply_to, m, targets=[m.rule1, m.x])

    def test_deprecation_warning_for_cuid_target(self):
        m = self.makeModel()
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            TransformationFactory('core.add_slack_variables').apply_to(m, targets=ComponentUID(m.rule3))
        self.assertRegex(out.getvalue(), 'DEPRECATED: In future releases ComponentUID targets will no longer be\nsupported in the core.add_slack_variables transformation. Specify\ntargets as a Constraint or list of Constraints.*')
        self.checkNonTargetCons(m)
        self.checkRule3(m)
        self.assertFalse(m.obj.active)
        self.checkTargetObj(m)
        transBlock = m.component('_core_add_slack_variables')
        self.checkTargetSlackVar(transBlock)

    def test_deprecation_warning_for_cuid_targets(self):
        m = self.makeModel()
        out = StringIO()
        with LoggingIntercept(out, 'pyomo.core'):
            TransformationFactory('core.add_slack_variables').apply_to(m, targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])
        self.assertRegex(out.getvalue(), 'DEPRECATED: In future releases ComponentUID targets will no longer be\nsupported in the core.add_slack_variables transformation. Specify\ntargets as a Constraint or list of Constraints.*')
        self.checkNonTargetCons(m)
        self.checkRule1(m)
        self.checkRule3(m)
        self.assertFalse(m.obj.active)
        self.checkTargetsObj(m)
        transBlock = m.component('_core_add_slack_variables')
        self.checkTargetSlackVars(transBlock)

    def test_transformed_constraints_sumexpression_body(self):
        m = self.makeModel()
        m.rule4 = Constraint(expr=inequality(5, m.x - 2 * m.y, 9))
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=m.rule4)
        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertEqual(c.lower, 5)
        self.assertEqual(c.upper, 9)
        assertExpressionsEqual(self, c.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.x)), EXPR.MonomialTermExpression((-2, m.y)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule4)), EXPR.MonomialTermExpression((-1, transBlock._slack_minus_rule4))]))

    def test_transformed_constraint_scalar_body(self):
        m = self.makeModel()
        m.p = Param(initialize=6, mutable=True)
        m.rule4 = Constraint(expr=m.p <= 9)
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule4])
        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertIsNone(c.lower)
        self.assertEqual(c.upper, 9)
        self.assertEqual(c.body.nargs(), 2)
        self.assertEqual(c.body.arg(0).value, 6)
        self.assertIs(c.body.arg(1).__class__, EXPR.MonomialTermExpression)
        self.assertEqual(c.body.arg(1).arg(0), -1)
        self.assertIs(c.body.arg(1).arg(1), transBlock._slack_minus_rule4)