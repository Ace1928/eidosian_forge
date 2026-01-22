import pyomo.environ as pyo
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging
def _check_var_bounds(m: _BlockData, too_large: float):
    vars_without_bounds = ComponentSet()
    vars_with_large_bounds = ComponentSet()
    for v in m.component_data_objects(pyo.Var, descend_into=True):
        if v.is_fixed():
            continue
        if v.lb is None or v.ub is None:
            vars_without_bounds.add(v)
        elif v.lb <= -too_large or v.ub >= too_large:
            vars_with_large_bounds.add(v)
    return (vars_without_bounds, vars_with_large_bounds)