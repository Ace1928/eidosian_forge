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
def add_cuts_components(self, model):
    config = self.config
    MindtPy = model.MindtPy_utils
    feas = MindtPy.feas_opt = Block()
    feas.deactivate()
    feas.feas_constraints = ConstraintList(doc='Feasibility Problem Constraints')
    lin = MindtPy.cuts = Block()
    lin.deactivate()
    lin.no_good_cuts = ConstraintList(doc='no-good cuts')
    if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
        feas.nl_constraint_set = RangeSet(len(MindtPy.nonlinear_constraint_list), doc='Integer index set over the nonlinear constraints.')
        feas.slack_var = Var(feas.nl_constraint_set, domain=NonNegativeReals, initialize=1)
    else:
        feas.slack_var = Var(domain=NonNegativeReals, initialize=1)
    if config.add_slack:
        lin.slack_vars = VarList(bounds=(0, config.max_slack), initialize=0, domain=NonNegativeReals)