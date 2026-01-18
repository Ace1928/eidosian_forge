from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def test_transformed_constraints_linear(self):
    m = models.makeTwoTermDisj_Nonlinear()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
    c1 = hull.get_transformed_constraints(m.d[1].c1)
    self.assertEqual(len(c1), 1)
    cons = c1[0]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
    ct.check_linear_coef(self, repn, m.d[1].indicator_var, 2)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(disjBlock[1].disaggregatedVars.x.lb, 0)
    self.assertEqual(disjBlock[1].disaggregatedVars.x.ub, 8)
    c2 = hull.get_transformed_constraints(m.d[1].c2)
    self.assertEqual(len(c2), 1)
    cons = c2[0]
    self.assertEqual(cons.lower, 0)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.w, 1)
    ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
    self.assertEqual(repn.constant, 0)
    self.assertEqual(disjBlock[1].disaggregatedVars.w.lb, 0)
    self.assertEqual(disjBlock[1].disaggregatedVars.w.ub, 7)
    c3 = hull.get_transformed_constraints(m.d[1].c3)
    self.assertEqual(len(c3), 2)
    cons = c3[0]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, -1)
    ct.check_linear_coef(self, repn, m.d[1].indicator_var, 1)
    self.assertEqual(repn.constant, 0)
    cons = c3[1]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, disjBlock[1].disaggregatedVars.x, 1)
    ct.check_linear_coef(self, repn, m.d[1].indicator_var, -3)
    self.assertEqual(repn.constant, 0)