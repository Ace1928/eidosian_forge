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
class TestSolverIterable(unittest.TestCase):
    """
    Test standardizer method for iterable of solvers,
    used to validate `backup_local_solvers` and `backup_global_solvers`
    arguments.
    """

    def setUp(self):
        SolverFactory.register(AVAILABLE_SOLVER_TYPE_NAME)(AvailableSolver)

    def tearDown(self):
        SolverFactory.unregister(AVAILABLE_SOLVER_TYPE_NAME)

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

    def test_solver_iterable_valid_str(self):
        """
        Test SolverIterable raises exception when str passed.
        """
        solver_str = AVAILABLE_SOLVER_TYPE_NAME
        standardizer_func = SolverIterable()
        solver_list = standardizer_func(solver_str)
        self.assertEqual(len(solver_list), 1, 'Standardized solver list is not of expected length')

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

    def test_solver_iterable_invalid_list(self):
        """
        Test SolverIterable raises exception if iterable contains
        at least one invalid object.
        """
        invalid_object = [AVAILABLE_SOLVER_TYPE_NAME, 2]
        standardizer_func = SolverIterable(solver_desc='backup solver')
        exc_str = 'Cannot cast object `2` to a Pyomo optimizer.*backup solver.*index 1.*got type int.*'
        with self.assertRaisesRegex(SolverNotResolvable, exc_str):
            standardizer_func(invalid_object)