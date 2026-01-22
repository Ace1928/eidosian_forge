from os.path import join, dirname, abspath
import json
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.block import IBlock
from pyomo.core import Suffix, Var, Constraint, Objective
from pyomo.opt import ProblemFormat, SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def all_models(arg=None):
    """
    Previously named "test_models" - renamed due to port to Pytest
    """
    if arg is None:
        return _test_models
    else:
        return _test_models[arg]