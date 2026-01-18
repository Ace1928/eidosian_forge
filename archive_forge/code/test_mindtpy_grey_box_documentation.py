from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
import pyomo.common.unittest as unittest
from pyomo.environ import SolverFactory, value, maximize
from pyomo.opt import TerminationCondition
from pyomo.common.dependencies import numpy_available, scipy_available
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
Test the outer approximation decomposition algorithm.