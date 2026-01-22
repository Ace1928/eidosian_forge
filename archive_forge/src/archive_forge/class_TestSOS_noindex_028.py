import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
class TestSOS_noindex_028(SOSProblem_nonindexed, unittest.TestCase):

    def test(self):
        error_triggered = False
        try:
            self.do_it(28)
        except NotImplementedError:
            error_triggered = True
        assert error_triggered