import logging
import unittest
from pyomo.core.base import ConcreteModel, Var, _VarData
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import ApplicationError
from pyomo.core.base.param import Param, _ParamData
from pyomo.contrib.pyros.config import (
from pyomo.contrib.pyros.util import ObjectiveType
from pyomo.opt import SolverFactory, SolverResults
from pyomo.contrib.pyros.uncertainty_sets import BoxSet
from pyomo.common.dependencies import numpy_available
def test_solver_resolvable_valid_solver_type(self):
    """
        Test solver resolvable class is valid for string
        type.
        """
    solver = SolverFactory(AVAILABLE_SOLVER_TYPE_NAME)
    standardizer_func = SolverResolvable()
    standardized_solver = standardizer_func(solver)
    self.assertIs(solver, standardized_solver, msg=f'Test solver {solver} and standardized solver {standardized_solver} are not identical.')