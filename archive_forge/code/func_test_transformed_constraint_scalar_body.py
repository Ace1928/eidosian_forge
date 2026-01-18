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
def test_transformed_constraint_scalar_body(self):
    m = self.makeModel()
    m.p = Param(initialize=6, mutable=True)
    m.rule4 = Constraint(expr=m.p <= 9)
    TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule4])
    transBlock = m._core_add_slack_variables
    c = m.rule4
    self.assertIsNone(c.lower)
    self.assertEqual(c.upper, 9)
    self.assertEqual(c.body.nargs(), 2)
    self.assertEqual(c.body.arg(0).value, 6)
    self.assertIs(c.body.arg(1).__class__, EXPR.MonomialTermExpression)
    self.assertEqual(c.body.arg(1).arg(0), -1)
    self.assertIs(c.body.arg(1).arg(1), transBlock._slack_minus_rule4)