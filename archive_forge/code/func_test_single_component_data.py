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
def test_single_component_data(self):
    """
        Test standardizer works for single component
        data-type entry.
        """
    mdl = ConcreteModel()
    mdl.v = Var([0, 1])
    standardizer_func = InputDataStandardizer(Var, _VarData)
    standardizer_input = mdl.v[0]
    standardizer_output = standardizer_func(standardizer_input)
    self.assertIsInstance(standardizer_output, list, msg=f'Standardized output should be of type list, but is of type {standardizer_output.__class__.__name__}.')
    self.assertEqual(len(standardizer_output), 1, msg='Length of standardizer output is not as expected.')
    self.assertIs(standardizer_output[0], mdl.v[0], msg=f'Entry {standardizer_output[0]} (id {id(standardizer_output[0])}) is not identical to input component data object {mdl.v[0]} (id {id(mdl.v[0])})')