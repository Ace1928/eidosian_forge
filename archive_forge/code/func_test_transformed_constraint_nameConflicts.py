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
def test_transformed_constraint_nameConflicts(self):
    m = models.makeTwoTermDisj_BlockOnDisj()
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    transBlock = m._pyomo_gdp_bigm_reformulation
    disjBlock = transBlock.relaxedDisjuncts
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    evil0 = bigm.get_transformed_constraints(m.evil[0].c)
    self.assertEqual(len(evil0), 1)
    self.assertIs(evil0[0].parent_block(), disjBlock[0])
    evil1 = bigm.get_transformed_constraints(m.evil[1].component('b.c'))
    self.assertEqual(len(evil1), 1)
    self.assertIs(evil1[0].parent_block(), disjBlock[1])
    evil1 = bigm.get_transformed_constraints(m.evil[1].b.c)
    self.assertEqual(len(evil1), 2)
    self.assertIs(evil1[0].parent_block(), disjBlock[1])
    self.assertIs(evil1[1].parent_block(), disjBlock[1])
    evil1 = bigm.get_transformed_constraints(m.evil[1].b.anotherblock.c)
    self.assertEqual(len(evil1), 1)
    self.assertIs(evil1[0].parent_block(), disjBlock[1])
    evil1 = bigm.get_transformed_constraints(m.evil[1].bb[1].c)
    self.assertEqual(len(evil1), 2)
    self.assertIs(evil1[0].parent_block(), disjBlock[1])
    self.assertIs(evil1[1].parent_block(), disjBlock[1])