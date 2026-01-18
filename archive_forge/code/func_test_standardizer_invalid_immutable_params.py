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
def test_standardizer_invalid_immutable_params(self):
    """
        Test standardizer raises exception when immutable
        Param object(s) passed.
        """
    standardizer_func = InputDataStandardizer(ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator)
    mdl = ConcreteModel()
    mdl.p = Param([0, 1], initialize=1)
    exc_str = 'Param object with name .*immutable'
    with self.assertRaisesRegex(ValueError, exc_str):
        standardizer_func(mdl.p)