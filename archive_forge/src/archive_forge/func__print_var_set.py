import pyomo.environ as pyo
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging
def _print_var_set(var_set):
    s = f'{'LB':>12}{'UB':>12}    Var\n'
    for v in var_set:
        v_lb, v_ub = _bounds_to_float(*v.bounds)
        s += f'{v_lb:>12.2e}{v_ub:>12.2e}    {str(v)}\n'
    s += '\n'
    return s