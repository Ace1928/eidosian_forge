from io import StringIO
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigBlock
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.config_options import _add_common_configs
from pyomo.contrib.gdpopt.create_oa_subproblems import (
from pyomo.contrib.gdpopt import __version__
from pyomo.contrib.gdpopt.util import (
from pyomo.core.base import Objective, value, minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.util.model_size import build_model_size_report
def _update_bounds_after_discrete_problem_solve(self, mip_termination, obj_expr, logger):
    if mip_termination is tc.optimal:
        self._update_bounds_after_solve('discrete', dual=value(obj_expr), logger=logger)
    elif mip_termination is tc.infeasible:
        self._update_dual_bound_to_infeasible()
    elif mip_termination is tc.feasible or tc.unbounded:
        pass
    else:
        raise DeveloperError('Unrecognized termination condition %s when updating the dual bound.' % mip_termination)