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
class TestLoggerType(unittest.TestCase):
    """
    Test logger type validator.
    """

    def test_logger_type(self):
        """
        Test logger type validator.
        """
        standardizer_func = LoggerType()
        mylogger = logging.getLogger('example')
        self.assertIs(standardizer_func(mylogger), mylogger, msg=f'{LoggerType.__name__} output not as expected')
        self.assertIs(standardizer_func(mylogger.name), mylogger, msg=f'{LoggerType.__name__} output not as expected')
        exc_str = 'A logger name must be a string'
        with self.assertRaisesRegex(Exception, exc_str):
            standardizer_func(2)