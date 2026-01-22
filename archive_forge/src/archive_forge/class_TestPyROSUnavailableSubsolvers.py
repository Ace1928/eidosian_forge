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
class TestPyROSUnavailableSubsolvers(unittest.TestCase):
    """
    Check that appropriate exceptionsa are raised if
    PyROS is invoked with unavailable subsolvers.
    """

    def test_pyros_unavailable_subsolver(self):
        """
        Test PyROS raises expected error message when
        unavailable subsolver is passed.
        """
        m = ConcreteModel()
        m.p = Param(range(3), initialize=0, mutable=True)
        m.z = Var([0, 1], initialize=0)
        m.con = Constraint(expr=m.z[0] + m.z[1] >= m.p[0])
        m.obj = Objective(expr=m.z[0] + m.z[1])
        pyros_solver = SolverFactory('pyros')
        exc_str = '.*Solver.*UnavailableSolver.*not available'
        with self.assertRaisesRegex(ValueError, exc_str):
            with LoggingIntercept(level=logging.ERROR) as LOG:
                pyros_solver.solve(model=m, first_stage_variables=[m.z[0]], second_stage_variables=[m.z[1]], uncertain_params=[m.p[0]], uncertainty_set=BoxSet([[0, 1]]), local_solver=SimpleTestSolver(), global_solver=UnavailableSolver())
        error_msgs = LOG.getvalue()[:-1]
        self.assertRegex(error_msgs, 'Output of `available\\(\\)` method.*global solver.*')

    @unittest.skipUnless(ipopt_available, 'IPOPT is not available.')
    def test_pyros_unavailable_backup_subsolver(self):
        """
        Test PyROS raises expected error message when
        unavailable backup subsolver is passed.
        """
        m = ConcreteModel()
        m.p = Param(range(3), initialize=0, mutable=True)
        m.z = Var([0, 1], initialize=0)
        m.con = Constraint(expr=m.z[0] + m.z[1] >= m.p[0])
        m.obj = Objective(expr=m.z[0] + m.z[1])
        pyros_solver = SolverFactory('pyros')
        with LoggingIntercept(level=logging.WARNING) as LOG:
            pyros_solver.solve(model=m, first_stage_variables=[m.z[0]], second_stage_variables=[m.z[1]], uncertain_params=[m.p[0]], uncertainty_set=BoxSet([[0, 1]]), local_solver=SolverFactory('ipopt'), global_solver=SolverFactory('ipopt'), backup_global_solvers=[UnavailableSolver()], bypass_global_separation=True)
        error_msgs = LOG.getvalue()[:-1]
        self.assertRegex(error_msgs, 'Output of `available\\(\\)` method.*backup global solver.*Removing from list.*')