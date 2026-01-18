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
def test_standardizer_invalid_str_passed(self):
    """
        Test standardizer raises exception as expected
        when input is of invalid type str.
        """
    standardizer_func = InputDataStandardizer(Var, _VarData)
    exc_str = 'Input object .*is not of valid component type.*'
    with self.assertRaisesRegex(TypeError, exc_str):
        standardizer_func('abcd')