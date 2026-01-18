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
def test_slack_vars_added(self):
    m = self.makeModel()
    TransformationFactory('core.add_slack_variables').apply_to(m)
    xblock = m.component('_core_add_slack_variables')
    self.assertIsInstance(xblock.component('_slack_minus_rule1'), Var)
    self.assertFalse(hasattr(xblock, '_slack_plus_rule1'))
    self.assertIsInstance(xblock.component('_slack_minus_rule2'), Var)
    self.assertIsInstance(xblock.component('_slack_plus_rule2'), Var)
    self.assertFalse(hasattr(xblock, '_slack_minus_rule3'))
    self.assertIsInstance(xblock.component('_slack_plus_rule3'), Var)
    self.assertEqual(xblock._slack_minus_rule1.bounds, (0, None))
    self.assertEqual(xblock._slack_minus_rule2.bounds, (0, None))
    self.assertEqual(xblock._slack_plus_rule2.bounds, (0, None))
    self.assertEqual(xblock._slack_plus_rule3.bounds, (0, None))