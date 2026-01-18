import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.nonconvex1 import Nonconvex1
from pyomo.contrib.mindtpy.tests.nonconvex2 import Nonconvex2
from pyomo.contrib.mindtpy.tests.nonconvex3 import Nonconvex3
from pyomo.contrib.mindtpy.tests.nonconvex4 import Nonconvex4
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.mcpp import pyomo_mcpp
Test the global outer approximation decomposition algorithm.