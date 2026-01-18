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
def test_transformed_constraint_mappings(self):
    m = models.makeTwoTermDisj_Nonlinear()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
    orig1 = m.d[0].c
    cons = hull.get_transformed_constraints(orig1)
    self.assertEqual(len(cons), 1)
    trans1 = cons[0]
    self.assertIs(trans1.parent_block(), disjBlock[0])
    self.assertIs(hull.get_src_constraint(trans1), orig1)
    orig1 = m.d[1].c1
    cons = hull.get_transformed_constraints(orig1)
    self.assertEqual(len(cons), 1)
    trans1 = cons[0]
    self.assertIs(trans1.parent_block(), disjBlock[1])
    self.assertIs(hull.get_src_constraint(trans1), orig1)
    orig2 = m.d[1].c2
    cons = hull.get_transformed_constraints(orig2)
    self.assertEqual(len(cons), 1)
    trans2 = cons[0]
    self.assertIs(trans1.parent_block(), disjBlock[1])
    self.assertIs(hull.get_src_constraint(trans2), orig2)
    orig3 = m.d[1].c3
    cons = hull.get_transformed_constraints(orig3)
    self.assertEqual(len(cons), 2)
    trans3 = cons[0]
    self.assertIs(hull.get_src_constraint(trans3), orig3)
    self.assertIs(trans3.parent_block(), disjBlock[1])
    trans32 = cons[1]
    self.assertIs(hull.get_src_constraint(trans32), orig3)
    self.assertIs(trans32.parent_block(), disjBlock[1])