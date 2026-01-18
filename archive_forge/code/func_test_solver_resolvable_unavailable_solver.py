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