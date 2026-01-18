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
def test_solver_iterable_unavailable_solver(self):
    """
        Test SolverIterable addresses unavailable solvers appropriately.
        """
    solvers = (AvailableSolver(), UnavailableSolver())
    standardizer_func = SolverIterable(require_available=True, filter_by_availability=True, solver_desc='example solver list')
    exc_str = 'Solver.*UnavailableSolver.* not available'
    with self.assertRaisesRegex(ApplicationError, exc_str):
        standardizer_func(solvers)
    with self.assertRaisesRegex(ApplicationError, exc_str):
        standardizer_func(solvers, filter_by_availability=False)
    standardized_solver_list = standardizer_func(solvers, filter_by_availability=True, require_available=False)
    self.assertEqual(len(standardized_solver_list), 1, msg='Length of filtered standardized solver list not as expected.')
    self.assertIs(standardized_solver_list[0], solvers[0], msg='Entry of filtered standardized solver list not as expected.')
    standardized_solver_list = standardizer_func(solvers, filter_by_availability=False, require_available=False)
    self.assertEqual(len(standardized_solver_list), 2, msg='Length of filtered standardized solver list not as expected.')
    self.assertEqual(standardized_solver_list, list(solvers), msg='Entry of filtered standardized solver list not as expected.')