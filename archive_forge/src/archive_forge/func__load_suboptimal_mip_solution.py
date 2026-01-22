from collections.abc import Iterable
import logging
import math
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.log import LogStream
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import Var, _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.core.staleflag import StaleFlagManager
import sys
def _load_suboptimal_mip_solution(self, vars_to_load, solution_number):
    if self.get_model_attr('NumIntVars') == 0 and self.get_model_attr('NumBinVars') == 0:
        raise ValueError('Cannot obtain suboptimal solutions for a continuous model')
    var_map = self._pyomo_var_to_solver_var_map
    ref_vars = self._referenced_variables
    original_solution_number = self.get_gurobi_param_info('SolutionNumber')[2]
    self.set_gurobi_param('SolutionNumber', solution_number)
    gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
    vals = self._solver_model.getAttr('Xn', gurobi_vars_to_load)
    res = ComponentMap()
    for var_id, val in zip(vars_to_load, vals):
        using_cons, using_sos, using_obj = ref_vars[var_id]
        if using_cons or using_sos or using_obj is not None:
            res[self._vars[var_id][0]] = val
    self.set_gurobi_param('SolutionNumber', original_solution_number)
    return res