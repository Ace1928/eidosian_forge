import pyomo.common.unittest as unittest
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.graphics import matplotlib_available, seaborn_available
from pyomo.opt import SolverFactory
def test_scenario_example(self):
    from pyomo.contrib.parmest.examples.semibatch import scenario_example
    scenario_example.main()