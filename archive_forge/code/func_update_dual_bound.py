import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def update_dual_bound(self, bound_value):
    """Update the dual bound.

        Call after solving relaxed problem, including relaxed NLP and MIP main problem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the dual bound.
        """
    if math.isnan(bound_value):
        return
    if self.objective_sense == minimize:
        self.dual_bound = max(bound_value, self.dual_bound)
        self.dual_bound_improved = self.dual_bound > self.dual_bound_progress[-1]
    else:
        self.dual_bound = min(bound_value, self.dual_bound)
        self.dual_bound_improved = self.dual_bound < self.dual_bound_progress[-1]
    self.dual_bound_progress.append(self.dual_bound)
    self.dual_bound_progress_time.append(get_main_elapsed_time(self.timing))
    if self.dual_bound_improved:
        self.update_gap()