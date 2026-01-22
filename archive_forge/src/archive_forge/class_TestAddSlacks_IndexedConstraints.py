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
class TestAddSlacks_IndexedConstraints(unittest.TestCase):

    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.S = Set(initialize=[1, 2, 3])
        m.x = Var(m.S)
        m.y = Var()

        def rule1_rule(m, s):
            return 2 * m.x[s] >= 4
        m.rule1 = Constraint(m.S, rule=rule1_rule)
        m.rule2 = Constraint(expr=m.y <= 6)
        m.obj = Objective(expr=sum((m.x[s] for s in m.S)) - m.y)
        return m

    def checkSlackVars_indexedtarget(self, transBlock):
        self.assertIsInstance(transBlock.component('_slack_plus_rule1[1]'), Var)
        self.assertIsInstance(transBlock.component('_slack_plus_rule1[2]'), Var)
        self.assertIsInstance(transBlock.component('_slack_plus_rule1[3]'), Var)
        self.assertIsNone(transBlock.component('_slack_minus_rule2'))

    def test_indexedtarget_only_create_slackvars_for_targets(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1])
        transBlock = m.component('_core_add_slack_variables')
        self.checkSlackVars_indexedtarget(transBlock)

    def test_indexedtarget_only_create_slackvars_for_targets_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1])
        transBlock = m2.component('_core_add_slack_variables')
        self.checkSlackVars_indexedtarget(transBlock)

    def checkRule2(self, m):
        cons = m.rule2
        self.assertEqual(cons.upper, 6)
        self.assertIsNone(cons.lower)
        self.assertIs(cons.body, m.y)

    def test_indexedtarget_nontarget_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1])
        self.checkRule2(m)

    def test_indexedtarget_nontarget_same_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1])
        self.checkRule2(m2)

    def checkTargetObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component('_slack_objective')
        self.assertIsInstance(obj, Objective)
        assertExpressionsEqual(self, obj.expr, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, transBlock.component('_slack_plus_rule1[1]'))), EXPR.MonomialTermExpression((1, transBlock.component('_slack_plus_rule1[2]'))), EXPR.MonomialTermExpression((1, transBlock.component('_slack_plus_rule1[3]')))]))

    def test_indexedtarget_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1])
        self.assertFalse(m.obj.active)
        self.checkTargetObj(m)

    def test_indexedtarget_objective_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1])
        self.assertFalse(m2.obj.active)
        self.checkTargetObj(m2)

    def checkTransformedRule1(self, m, i):
        c = m.rule1[i]
        self.assertEqual(c.lower, 4)
        self.assertIsNone(c.upper)
        assertExpressionsEqual(self, c.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((2, m.x[i])), EXPR.MonomialTermExpression((1, m._core_add_slack_variables.component('_slack_plus_rule1[%s]' % i)))]))

    def test_indexedtarget_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1])
        for i in [1, 2, 3]:
            self.checkTransformedRule1(m, i)

    def test_indexedtarget_targets_transformed_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=m.rule1)
        for i in [1, 2, 3]:
            self.checkTransformedRule1(m2, i)

    def checkSlackVars_constraintDataTarget(self, transBlock):
        self.assertIsInstance(transBlock.component('_slack_plus_rule1[2]'), Var)
        self.assertIsNone(transBlock.component('_slack_plus_rule1[1]'))
        self.assertIsNone(transBlock.component('_slack_plus_rule1[3]'))
        self.assertIsNone(transBlock.component('_slack_minus_rule2'))

    def test_ConstraintDatatarget_only_add_slackvars_for_targets(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1[2]])
        transBlock = m._core_add_slack_variables
        self.checkSlackVars_constraintDataTarget(transBlock)

    def test_ConstraintDatatarget_only_add_slackvars_for_targets_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=m.rule1[2])
        transBlock = m2._core_add_slack_variables
        self.checkSlackVars_constraintDataTarget(transBlock)

    def checkUntransformedRule1(self, m, i):
        c = m.rule1[i]
        self.assertEqual(c.lower, 4)
        self.assertIsNone(c.upper)
        self.assertEqual(c.body.arg(0), 2)
        self.assertIs(c.body.arg(1), m.x[i])

    def test_ConstraintDatatarget_nontargets_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1[2]])
        self.checkUntransformedRule1(m, 1)
        self.checkUntransformedRule1(m, 3)
        self.checkRule2(m)

    def test_ConstraintDatatarget_nontargets_same_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1[2]])
        self.checkUntransformedRule1(m2, 1)
        self.checkUntransformedRule1(m2, 3)
        self.checkRule2(m2)

    def test_ConstraintDatatarget_target_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1[2]])
        self.checkTransformedRule1(m, 2)

    def test_ConstraintDatatarget_target_transformed_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1[2]])
        self.checkTransformedRule1(m2, 2)

    def checkConstraintDataObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component('_slack_objective')
        self.assertIsInstance(obj, Objective)
        self.assertIs(obj.expr, transBlock.component('_slack_plus_rule1[2]'))

    def test_ConstraintDatatarget_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1[2]])
        self.assertFalse(m.obj.active)
        self.checkConstraintDataObj(m)

    def test_ConstraintDatatarget_objective_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1[2]])
        self.assertFalse(m2.obj.active)
        self.checkConstraintDataObj(m2)