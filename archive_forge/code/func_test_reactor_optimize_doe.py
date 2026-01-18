from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
from pyomo.opt import SolverFactory
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, 'Numpy is not available')
def test_reactor_optimize_doe(self):
    from pyomo.contrib.doe.examples import reactor_optimize_doe
    reactor_optimize_doe.main()