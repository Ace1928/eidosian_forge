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
def test_transformed_constraints_on_block(self):
    m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    transBlock = m.component('_pyomo_gdp_bigm_reformulation')
    self.assertIsInstance(transBlock, Block)
    disjBlock = transBlock.component('relaxedDisjuncts')
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    cons11 = bigm.get_transformed_constraints(m.disjunct[0].c[1])
    self.assertEqual(len(cons11), 1)
    cons11_lb = cons11[0]
    self.assertIsInstance(cons11_lb.parent_component(), Constraint)
    self.assertTrue(cons11_lb.active)
    cons12 = bigm.get_transformed_constraints(m.disjunct[0].c[2])
    self.assertEqual(len(cons12), 1)
    cons12_lb = cons12[0]
    self.assertIsInstance(cons12_lb.parent_component(), Constraint)
    self.assertTrue(cons12_lb.active)
    cons21 = bigm.get_transformed_constraints(m.disjunct[1].c[1])
    self.assertEqual(len(cons21), 2)
    cons21_lb = cons21[0]
    cons21_ub = cons21[1]
    self.assertIsInstance(cons21_lb.parent_component(), Constraint)
    self.assertIsInstance(cons21_ub.parent_component(), Constraint)
    self.assertTrue(cons21_lb.active)
    self.assertTrue(cons21_ub.active)
    cons22 = bigm.get_transformed_constraints(m.disjunct[1].c[2])
    self.assertEqual(len(cons22), 2)
    cons22_lb = cons22[0]
    cons22_ub = cons22[1]
    self.assertIsInstance(cons22_lb.parent_component(), Constraint)
    self.assertIsInstance(cons22_ub.parent_component(), Constraint)
    self.assertTrue(cons22_lb.active)
    self.assertTrue(cons22_ub.active)