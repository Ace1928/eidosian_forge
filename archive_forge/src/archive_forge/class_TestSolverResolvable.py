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
class TestSolverResolvable(unittest.TestCase):
    """
    Test PyROS standardizer for solver-type objects.
    """

    def setUp(self):
        SolverFactory.register(AVAILABLE_SOLVER_TYPE_NAME)(AvailableSolver)

    def tearDown(self):
        SolverFactory.unregister(AVAILABLE_SOLVER_TYPE_NAME)

    def test_solver_resolvable_valid_str(self):
        """
        Test solver resolvable class is valid for string
        type.
        """
        solver_str = AVAILABLE_SOLVER_TYPE_NAME
        standardizer_func = SolverResolvable()
        solver = standardizer_func(solver_str)
        expected_solver_type = type(SolverFactory(solver_str))
        self.assertIsInstance(solver, type(SolverFactory(solver_str)), msg=f'SolverResolvable object should be of type {expected_solver_type.__name__}, but got object of type {solver.__class__.__name__}.')

    def test_solver_resolvable_valid_solver_type(self):
        """
        Test solver resolvable class is valid for string
        type.
        """
        solver = SolverFactory(AVAILABLE_SOLVER_TYPE_NAME)
        standardizer_func = SolverResolvable()
        standardized_solver = standardizer_func(solver)
        self.assertIs(solver, standardized_solver, msg=f'Test solver {solver} and standardized solver {standardized_solver} are not identical.')

    def test_solver_resolvable_invalid_type(self):
        """
        Test solver resolvable object raises expected
        exception when invalid entry is provided.
        """
        invalid_object = 2
        standardizer_func = SolverResolvable(solver_desc='local solver')
        exc_str = 'Cannot cast object `2` to a Pyomo optimizer.*local solver.*got type int.*'
        with self.assertRaisesRegex(SolverNotResolvable, exc_str):
            standardizer_func(invalid_object)

    def test_solver_resolvable_unavailable_solver(self):
        """
        Test solver standardizer fails in event solver is
        unavailable.
        """
        unavailable_solver = UnavailableSolver()
        standardizer_func = SolverResolvable(solver_desc='local solver', require_available=True)
        exc_str = 'Solver.*UnavailableSolver.*not available'
        with self.assertRaisesRegex(ApplicationError, exc_str):
            with LoggingIntercept(level=logging.ERROR) as LOG:
                standardizer_func(unavailable_solver)
        error_msgs = LOG.getvalue()[:-1]
        self.assertRegex(error_msgs, 'Output of `available\\(\\)` method.*local solver.*')