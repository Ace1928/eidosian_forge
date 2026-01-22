import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
class TestSOS_indexed_016(SOSProblem_indexed, unittest.TestCase):

    def test(self):
        error_triggered = False
        try:
            self.do_it(16)
        except NotImplementedError:
            error_triggered = True
        assert error_triggered