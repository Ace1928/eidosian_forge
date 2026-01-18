import logging
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.contrib.trustregion.TRF import trust_region_method, _trf_config
def test_solver(self):
    log_OUTPUT = StringIO()
    print_OUTPUT = StringIO()
    sys.stdout = print_OUTPUT
    with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
        result = trust_region_method(self.m, self.decision_variables, self.ext_fcn_surrogate_map_rule, self.config)
    sys.stdout = sys.__stdout__
    self.assertIn('Iteration 0', log_OUTPUT.getvalue())
    self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
    self.assertEqual(result.name, self.m.name)
    self.assertNotEqual(value(result.obj), value(self.m.obj))