import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
def test_or_constraints(self):
    m = models.makeTwoTermDisj()
    m.disjunction.xor = False
    TransformationFactory('gdp.binary_multiplication').apply_to(m)
    orcons = m._pyomo_gdp_binary_multiplication_reformulation.component('disjunction_xor')
    self.assertIsInstance(orcons, Constraint)
    assertExpressionsEqual(self, orcons.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.d[0].binary_indicator_var)), EXPR.MonomialTermExpression((1, m.d[1].binary_indicator_var))]))
    self.assertEqual(orcons.lower, 1)
    self.assertIsNone(orcons.upper)