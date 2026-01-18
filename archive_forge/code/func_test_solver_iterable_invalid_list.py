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