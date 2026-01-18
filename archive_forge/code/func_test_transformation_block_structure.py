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
def test_transformation_block_structure(self):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.bigm').apply_to(m)
    transBlock = m.disjunction.algebraic_constraint.parent_block()
    pairs = [(0, [m.simpledisjunct.innerdisjunct0.c]), (1, [m.simpledisjunct.innerdisjunct1.c]), (2, []), (5, [m.disjunct[0].c]), (2, [m.disjunct[1].innerdisjunct[0].c]), (3, [m.disjunct[1].innerdisjunct[1].c]), (6, [])]
    self.check_disjunction_transformation_block_structure(transBlock, pairs)
    self.assertIsInstance(transBlock.component('disjunction_xor'), Constraint)