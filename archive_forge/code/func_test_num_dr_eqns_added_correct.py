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
@unittest.skipIf(not scipy_available, 'Scipy is not available.')
def test_num_dr_eqns_added_correct(self):
    """
        Check that number of DR equality constraints added
        by constraint declaration routines matches the number
        of second-stage variables in the model.
        """
    model_data = ROSolveResults()
    model_data.working_model = m = self.make_simple_test_model()
    m.decision_rule_var_0 = Var([0], initialize=0)
    m.decision_rule_var_1 = Var([0], initialize=0)
    m.util.decision_rule_vars = [m.decision_rule_var_0, m.decision_rule_var_1]
    config = Bunch()
    config.decision_rule_order = 0
    add_decision_rule_constraints(model_data=model_data, config=config)
    self.assertEqual(len(m.util.decision_rule_eqns), len(m.util.second_stage_variables), msg='The number of decision rule constraints added to model should equalthe number of control variables in the model.')