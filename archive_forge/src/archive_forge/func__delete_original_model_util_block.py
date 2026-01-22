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
def _delete_original_model_util_block(self):
    """For cleaning up after a solve--we want the original model to be
        untouched except for the solution being loaded"""
    blk = self.original_util_block
    if blk is not None:
        blk.parent_block().del_component(blk)
    if self.original_obj is not None:
        self.original_obj.activate()
    if self._dummy_obj is not None:
        self._dummy_obj.parent_block().del_component(self._dummy_obj)
        self._dummy_obj = None