import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class testAddScenarioToMaster(unittest.TestCase):

    def test_add_scenario_to_master(self):
        working_model = ConcreteModel()
        working_model.p = Param([1, 2], initialize=0, mutable=True)
        working_model.x = Var()
        model_data = MasterProblemData()
        model_data.working_model = working_model
        model_data.timing = None
        master_data = initial_construct_master(model_data)
        master_data.master_model.scenarios[0, 0].transfer_attributes_from(working_model.clone())
        master_data.master_model.scenarios[0, 0].util = Block()
        master_data.master_model.scenarios[0, 0].util.first_stage_variables = [master_data.master_model.scenarios[0, 0].x]
        master_data.master_model.scenarios[0, 0].util.uncertain_params = [master_data.master_model.scenarios[0, 0].p[1], master_data.master_model.scenarios[0, 0].p[2]]
        add_scenario_to_master(master_data, violations=[1, 1])
        self.assertEqual(len(master_data.master_model.scenarios), 2, msg='Scenario not added to master correctly. Expected 2 scenarios.')