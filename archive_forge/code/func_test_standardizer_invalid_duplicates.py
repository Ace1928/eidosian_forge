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
def test_standardizer_invalid_duplicates(self):
    """
        Test standardizer raises exception if input contains duplicates
        and duplicates are not allowed.
        """
    mdl = ConcreteModel()
    mdl.v = Var([0, 1])
    mdl.x = Var(['a', 'b'])
    standardizer_func = InputDataStandardizer(Var, _VarData, allow_repeats=False)
    exc_str = 'Standardized.*list.*contains duplicate entries\\.'
    with self.assertRaisesRegex(ValueError, exc_str):
        standardizer_func([mdl.x, mdl.v, mdl.x])