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
def test_standardizer_valid_mutable_params(self):
    """
        Test Param-like standardizer works as expected for sequence
        of valid mutable Param objects.
        """
    mdl = ConcreteModel()
    mdl.p1 = Param([0, 1], initialize=0, mutable=True)
    mdl.p2 = Param(['a', 'b'], initialize=1, mutable=True)
    standardizer_func = InputDataStandardizer(ctype=Param, cdatatype=_ParamData, ctype_validator=mutable_param_validator)
    standardizer_input = [mdl.p1[0], mdl.p2]
    standardizer_output = standardizer_func(standardizer_input)
    expected_standardizer_output = [mdl.p1[0], mdl.p2['a'], mdl.p2['b']]
    self.assertIsInstance(standardizer_output, list, msg=f'Standardized output should be of type list, but is of type {standardizer_output.__class__.__name__}.')
    self.assertEqual(len(standardizer_output), len(expected_standardizer_output), msg='Length of standardizer output is not as expected.')
    enum_zip = enumerate(zip(expected_standardizer_output, standardizer_output))
    for idx, (input, output) in enum_zip:
        self.assertIs(input, output, msg=f'Entry {input} (id {id(input)}) is not identical to input component data object {output} (id {id(output)})')