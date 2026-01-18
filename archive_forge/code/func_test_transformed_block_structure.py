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
def test_transformed_block_structure(self):
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.bigm').apply_to(m)
    transBlock = m.component('_pyomo_gdp_bigm_reformulation')
    self.assertIsInstance(transBlock, Block)
    disjBlock = transBlock.relaxedDisjuncts
    self.assertEqual(len(disjBlock), 8)
    for i, j in self.pairs:
        self.assertEqual(len(disjBlock[j].component_map(Constraint)), 1)