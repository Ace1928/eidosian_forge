import logging
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.contrib.trustregion.TRF import trust_region_method, _trf_config
def test_update_kwdval(self):
    self.TRF = SolverFactory('trustregion')
    self.assertEqual(self.TRF.config.trust_radius, 1.0)
    self.TRF.config.trust_radius = 4.0
    solve_status = self.try_solve()
    self.assertTrue(solve_status)
    self.assertEqual(self.TRF.config.trust_radius, 4.0)