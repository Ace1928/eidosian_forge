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
def test_solver_iterable_valid_list(self):
    """
        Test solver type standardizer works for list of valid
        objects castable to solver.
        """
    solver_list = [AVAILABLE_SOLVER_TYPE_NAME, SolverFactory(AVAILABLE_SOLVER_TYPE_NAME)]
    expected_solver_types = [AvailableSolver] * 2
    standardizer_func = SolverIterable()
    standardized_solver_list = standardizer_func(solver_list)
    for idx, standardized_solver in enumerate(standardized_solver_list):
        self.assertIsInstance(standardized_solver, expected_solver_types[idx], msg=f'Standardized solver {standardized_solver} (index {idx}) expected to be of type {expected_solver_types[idx].__name__}, but is of type {standardized_solver.__class__.__name__}')
    self.assertIs(standardized_solver_list[1], solver_list[1], msg=f'Test solver {solver_list[1]} and standardized solver {standardized_solver_list[1]} should be identical.')