import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
Test the LP/NLP decomposition algorithm.