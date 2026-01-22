import pyomo.environ as pyo
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging
def _check_coefficients(comp, expr, too_large, too_small, largs_coef_map, small_coef_map):
    ders = reverse_sd(expr)
    for _v, _der in ders.items():
        if isinstance(_v, _GeneralVarData):
            if _v.is_fixed():
                continue
            der_lb, der_ub = compute_bounds_on_expr(_der)
            der_lb, der_ub = _bounds_to_float(der_lb, der_ub)
            if der_lb <= -too_large or der_ub >= too_large:
                if comp not in largs_coef_map:
                    largs_coef_map[comp] = list()
                largs_coef_map[comp].append((_v, der_lb, der_ub))
            if abs(der_lb) <= too_small and abs(der_ub) < too_small:
                if der_lb != 0 or der_ub != 0:
                    if comp not in small_coef_map:
                        small_coef_map[comp] = list()
                    small_coef_map[comp].append((_v, der_lb, der_ub))