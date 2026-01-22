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
def _transfer_incumbent_to_original_model(self, logger):
    StaleFlagManager.mark_all_as_stale(delayed=False)
    if self.incumbent_boolean_soln is None:
        assert self.incumbent_continuous_soln is None
        logger.info('No feasible solutions found.')
        return
    for var, soln in zip(self.original_util_block.algebraic_variable_list, self.incumbent_continuous_soln):
        var.set_value(soln, skip_validation=True)
    for var, soln in zip(self.original_util_block.boolean_variable_list, self.incumbent_boolean_soln):
        if soln is None:
            var.set_value(soln, skip_validation=True)
        elif soln > 0.5:
            var.set_value(True)
        else:
            var.set_value(False)
    StaleFlagManager.mark_all_as_stale(delayed=True)