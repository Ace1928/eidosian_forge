import pyomo.environ as pyo
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging
def _print_coefficients(comp_map):
    s = ''
    for c, der_bounds in comp_map.items():
        s += str(c)
        s += '\n'
        s += f'    {'Coef LB':>12}{'Coef UB':>12}    Var\n'
        for v, der_lb, der_ub in der_bounds:
            s += f'    {der_lb:>12.2e}{der_ub:>12.2e}    {str(v)}\n'
        s += '\n'
    return s