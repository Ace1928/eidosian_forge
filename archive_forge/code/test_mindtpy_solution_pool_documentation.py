from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP2_simple import SimpleMINLP as SimpleMINLP2
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.environ import SolverFactory, value, maximize
from pyomo.opt import TerminationCondition
Test the outer approximation decomposition algorithm.