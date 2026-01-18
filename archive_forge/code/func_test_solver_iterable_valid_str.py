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
def test_solver_iterable_valid_str(self):
    """
        Test SolverIterable raises exception when str passed.
        """
    solver_str = AVAILABLE_SOLVER_TYPE_NAME
    standardizer_func = SolverIterable()
    solver_list = standardizer_func(solver_str)
    self.assertEqual(len(solver_list), 1, 'Standardized solver list is not of expected length')