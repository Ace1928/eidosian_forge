from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
class ScalarDisjIndexedConstraints(unittest.TestCase, CommonTests):

    def setUp(self):
        random.seed(666)

    def test_do_not_transform_deactivated_constraintDatas(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 30
        m.b.simpledisj1.c[1].deactivate()
        bigm = TransformationFactory('gdp.bigm')
        bigm.apply_to(m)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.ERROR):
            self.assertRaisesRegex(KeyError, '.*b.simpledisj1.c\\[1\\]', bigm.get_transformed_constraints, m.b.simpledisj1.c[1])
        self.assertRegex(log.getvalue(), ".*Constraint 'b.simpledisj1.c\\[1\\]' has not been transformed.")
        cons_list = bigm.get_transformed_constraints(m.b.simpledisj1.c[2])
        self.assertEqual(len(cons_list), 2)
        lb = cons_list[0]
        ub = cons_list[1]
        self.assertIsInstance(lb, constraint._GeneralConstraintData)
        self.assertIsInstance(ub, constraint._GeneralConstraintData)

    def checkMs(self, m, disj1c1lb, disj1c1ub, disj1c2lb, disj1c2ub, disj2c1ub, disj2c2ub):
        bigm = TransformationFactory('gdp.bigm')
        m_values = bigm.get_all_M_values_by_constraint(m)
        c = bigm.get_transformed_constraints(m.b.simpledisj1.c[1])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1lb)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c1lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c1ub)
        self.assertIn(m.b.simpledisj1.c[1], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj1.c[1]][0], disj1c1lb)
        self.assertEqual(m_values[m.b.simpledisj1.c[1]][1], disj1c1ub)
        c = bigm.get_transformed_constraints(m.b.simpledisj1.c[2])
        self.assertEqual(len(c), 2)
        lb = c[0]
        ub = c[1]
        repn = generate_standard_repn(lb.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2lb)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c2lb)
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj1.indicator_var, disj1c2ub)
        self.assertIn(m.b.simpledisj1.c[2], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj1.c[2]][0], disj1c2lb)
        self.assertEqual(m_values[m.b.simpledisj1.c[2]][1], disj1c2ub)
        c = bigm.get_transformed_constraints(m.b.simpledisj2.c[1])
        self.assertEqual(len(c), 1)
        ub = c[0]
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c1ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj2.indicator_var, disj2c1ub)
        self.assertIn(m.b.simpledisj2.c[1], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj2.c[1]][1], disj2c1ub)
        self.assertIsNone(m_values[m.b.simpledisj2.c[1]][0])
        c = bigm.get_transformed_constraints(m.b.simpledisj2.c[2])
        self.assertEqual(len(c), 1)
        ub = c[0]
        repn = generate_standard_repn(ub.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c2ub)
        ct.check_linear_coef(self, repn, m.b.simpledisj2.indicator_var, disj2c2ub)
        self.assertIn(m.b.simpledisj2.c[2], m_values.keys())
        self.assertEqual(m_values[m.b.simpledisj2.c[2]][1], disj2c2ub)
        self.assertIsNone(m_values[m.b.simpledisj2.c[2]][0])
        self.assertEqual(len(m_values), 4)

    def test_suffix_M_constraintData_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj1.c[1]] = 15
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -15, 15, -30, 30, 30, 30)

    def test_suffix_M_indexedConstraint_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj2.c] = 15
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -30, 30, -30, 30, 15, 15)

    def test_suffix_M_constraintData_on_simpleDisjunct(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c[2]] = (-14, 13)
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -65, 65, -14, 13, 65, 65)

    def test_suffix_M_indexedConstraint_on_simpleDisjunct(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c] = (-14, 13)
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -14, 13, -14, 13, 65, 65)

    def test_unbounded_var_m_estimation_err(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        self.assertRaisesRegex(GDP_Error, "Cannot estimate M for unbounded expressions.\\n\\t\\(found while processing constraint 'b.simpledisj1.c\\[1\\]'\\). Please specify a value of M or ensure all variables that appear in the constraint are bounded.", TransformationFactory('gdp.bigm').apply_to, m)

    def test_create_using(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 100
        self.diff_apply_to_and_create_using(m)