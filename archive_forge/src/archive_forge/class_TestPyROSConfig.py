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
class TestPyROSConfig(unittest.TestCase):
    """
    Test PyROS ConfigDict behaves as expected.
    """
    CONFIG = pyros_config()

    def test_config_objective_focus(self):
        """
        Test config parses objective focus as expected.
        """
        config = self.CONFIG()
        for obj_focus_name in ['nominal', 'worst_case']:
            config.objective_focus = obj_focus_name
            self.assertEqual(config.objective_focus, ObjectiveType[obj_focus_name], msg='Objective focus not set as expected.')
        for obj_focus in ObjectiveType:
            config.objective_focus = obj_focus
            self.assertEqual(config.objective_focus, obj_focus, msg='Objective focus not set as expected.')
        invalid_focus = 'test_example'
        exc_str = f'.*{invalid_focus!r} is not a valid ObjectiveType'
        with self.assertRaisesRegex(ValueError, exc_str):
            config.objective_focus = invalid_focus