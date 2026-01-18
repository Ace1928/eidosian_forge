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
def test_warning_for_crazy_bigm_args(self):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    out = StringIO()
    bigM = ComponentMap({m: 100, m.b.disjunct[1].c: 13})
    bigM[m.a] = 34
    with LoggingIntercept(out, 'pyomo.gdp.bigm'):
        TransformationFactory('gdp.bigm').apply_to(m, bigM=bigM)
    self.checkMs(m, -100, 100, 13, -100, 100)
    self.assertEqual(out.getvalue(), 'Unused arguments in the bigM map! These arguments were not used by the transformation:\n\ta\n\n')